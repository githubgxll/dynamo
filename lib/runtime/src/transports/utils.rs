// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{future::Future, ops::Deref, sync::Arc};

use anyhow::Result;

/// Keeps a dedicated transport executor alive while any client still uses it.
/// The executor itself is owned and dropped by its dedicated OS thread, never
/// by a caller running inside another Tokio runtime.
#[derive(Clone, Debug)]
pub struct TransportRuntime {
    handle: tokio::runtime::Handle,
    _lease: Arc<tokio::sync::oneshot::Sender<()>>,
}

impl Deref for TransportRuntime {
    type Target = tokio::runtime::Handle;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

pub async fn build_in_runtime<
    T: Send + Sync + 'static,
    F: Future<Output = Result<T>> + Send + 'static,
>(
    f: F,
    num_threads: usize,
) -> Result<(T, TransportRuntime)> {
    let (mut tx, rx) = tokio::sync::oneshot::channel();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_threads)
        .enable_all()
        .build()?;

    std::thread::spawn(move || {
        runtime.block_on(async {
            let result = tokio::select! {
                biased;
                _ = tx.closed() => return,
                result = f => result,
            };
            match result {
                Ok(value) => {
                    let (lease, released) = tokio::sync::oneshot::channel();
                    let executor = TransportRuntime {
                        handle: runtime.handle().clone(),
                        _lease: Arc::new(lease),
                    };
                    // Dropping an unread result also releases the lease. A
                    // successful receiver keeps it only as long as its clients.
                    if tx.send(Ok((value, executor))).is_ok() {
                        let _ = released.await;
                    }
                }
                Err(error) => {
                    let _ = tx.send(Err(error));
                }
            }
        });
        runtime.shutdown_background();
    });

    rx.await?
}

#[cfg(test)]
mod tests {
    use super::*;

    // Regression: cancelling startup must drop pending connection work on the
    // dedicated transport thread instead of leaving it running indefinitely.
    #[tokio::test]
    async fn cancelled_build_drops_pending_connection() {
        let dropped = tokio_util::sync::CancellationToken::new();
        let guard = dropped.clone().drop_guard();
        let (started, waiting) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(build_in_runtime(
            async move {
                let _guard = guard;
                started.send(()).unwrap();
                std::future::pending::<Result<()>>().await
            },
            1,
        ));
        waiting.await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        tokio::time::timeout(std::time::Duration::from_secs(5), dropped.cancelled())
            .await
            .expect("pending transport initialization must be dropped");
    }

    #[tokio::test]
    async fn cancelled_build_drops_sent_but_unread_runtime() {
        use std::task::{Context, Wake, Waker};

        struct ResultReady(tokio::sync::Notify);
        impl Wake for ResultReady {
            fn wake(self: Arc<Self>) {
                self.0.notify_one();
            }
        }

        let runtime_dropped = tokio_util::sync::CancellationToken::new();
        let guard = runtime_dropped.clone().drop_guard();
        let (release, proceed) = tokio::sync::oneshot::channel();
        let mut construction = Box::pin(build_in_runtime(
            async move {
                proceed.await.unwrap();
                // This task belongs to the dedicated runtime, not the caller.
                // Its guard is released only when that runtime shuts down.
                tokio::spawn(async move {
                    let _guard = guard;
                    std::future::pending::<()>().await;
                });
                Ok(())
            },
            1,
        ));
        let ready = Arc::new(ResultReady(tokio::sync::Notify::new()));
        let waker = Waker::from(ready.clone());
        assert!(
            construction
                .as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_pending()
        );
        release.send(()).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(5), ready.0.notified())
            .await
            .expect("producer queues the result and wakes its receiver");
        // Never poll again: the successful result stays in the oneshot channel
        // until cancellation drops its value and executor lease.
        assert!(!runtime_dropped.is_cancelled());
        drop(construction);
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            runtime_dropped.cancelled(),
        )
        .await
        .expect("unread result must not retain the dedicated runtime");
    }

    #[tokio::test]
    async fn successful_runtime_stops_after_last_client_release() {
        let dropped = tokio_util::sync::CancellationToken::new();
        let guard = dropped.clone().drop_guard();
        let (_, runtime) = build_in_runtime(
            async move {
                tokio::spawn(async move {
                    let _guard = guard;
                    std::future::pending::<()>().await;
                });
                Ok(())
            },
            1,
        )
        .await
        .unwrap();
        let other_client = runtime.clone();
        drop(runtime);
        assert_eq!(other_client.spawn(async { 42 }).await.unwrap(), 42);
        assert!(!dropped.is_cancelled());
        drop(other_client);
        tokio::time::timeout(std::time::Duration::from_secs(5), dropped.cancelled())
            .await
            .expect("last client release must stop transport tasks");
    }
}
