#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from dingo.frontend.main import main

if __name__ == "__main__":
    _maybe_preload_jemalloc()

    # Import the runtime only after configuring the allocator.
    from dingo.frontend.main import main

    main()
