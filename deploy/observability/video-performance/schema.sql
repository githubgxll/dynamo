-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- Independent v2 database only: do not apply over the legacy collector database.
CREATE TABLE IF NOT EXISTS collector_schema (version integer PRIMARY KEY);
INSERT INTO collector_schema VALUES (2) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS video_performance (
  cluster_id text NOT NULL, namespace text NOT NULL,
  deployment text NOT NULL, task_id text NOT NULL, attempt integer NOT NULL,
  completed_at timestamptz NOT NULL, model text NOT NULL, pool text NOT NULL,
  status text NOT NULL, configuration_revision text NOT NULL,
  environment_revision text NOT NULL, schema_id text NOT NULL,
  source text NOT NULL, seconds double precision, width integer, height integer,
  fps double precision, steps integer, sound boolean, input_type text NOT NULL,
  image_count integer, video_count integer, audio_count integer,
  image_pixels bigint, image_bytes bigint, video_seconds double precision,
  video_bytes bigint, video_frames bigint, audio_seconds double precision,
  audio_bytes bigint, e2e_s double precision, queue_s double precision,
  worker_queue_s double precision, execution_s double precision,
  finalize_s double precision, stages jsonb NOT NULL, details jsonb NOT NULL,
  PRIMARY KEY (cluster_id, namespace, deployment, task_id, attempt)
);
CREATE INDEX IF NOT EXISTS video_performance_time ON video_performance(completed_at);
CREATE INDEX IF NOT EXISTS video_performance_filter ON video_performance(pool, seconds, steps, input_type);
CREATE INDEX IF NOT EXISTS video_performance_namespace_time ON video_performance(namespace, completed_at);
CREATE OR REPLACE VIEW video_performance_stages AS
SELECT p.*, s.key AS stage, (s.value::text)::double precision AS stage_s
FROM video_performance p CROSS JOIN LATERAL jsonb_each(p.stages) s;
CREATE TABLE IF NOT EXISTS performance_environments (
  revision text PRIMARY KEY, observed_at timestamptz NOT NULL DEFAULT now(), snapshot jsonb NOT NULL
);
CREATE TABLE IF NOT EXISTS collector_sources (
  cluster_id text NOT NULL, namespace text NOT NULL, pod_uid text NOT NULL,
  container text NOT NULL, restart integer NOT NULL, pod text NOT NULL,
  cursor_time text, cursor_count integer NOT NULL DEFAULT 0,
  last_poll timestamptz, last_error text, inserted_events bigint NOT NULL DEFAULT 0,
  invalid_events bigint NOT NULL DEFAULT 0, truncated boolean NOT NULL DEFAULT false,
  PRIMARY KEY (cluster_id, namespace, pod_uid, container, restart)
);
CREATE TABLE IF NOT EXISTS performance_collector_health (
  id integer PRIMARY KEY CHECK (id = 1), last_poll timestamptz,
  last_event timestamptz, last_error text, inserted_events bigint NOT NULL DEFAULT 0,
  details jsonb NOT NULL DEFAULT '{}'
);
INSERT INTO performance_collector_health(id) VALUES(1) ON CONFLICT DO NOTHING;

-- Additive HTTP-pull state; samples and cursors commit in the same transaction.
CREATE TABLE IF NOT EXISTS performance_http_sources (
  target_id text PRIMARY KEY, binding text NOT NULL, instance_id text,
  cursor text, last_poll timestamptz, last_success timestamptz, last_error text,
  inserted_events bigint NOT NULL DEFAULT 0, invalid_events bigint NOT NULL DEFAULT 0,
  gap_events bigint NOT NULL DEFAULT 0, instance_changes bigint NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS performance_http_rejections (
  target_id text NOT NULL, instance_id text NOT NULL, seq bigint NOT NULL,
  reason text NOT NULL, observed_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (target_id, instance_id, seq)
);
