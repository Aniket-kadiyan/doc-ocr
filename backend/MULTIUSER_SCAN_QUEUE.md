# Multi-user scan queue

This milestone makes the existing in-process auto-balloon queue safe and
understandable when several authenticated LAN users submit work to one QTracX
backend.

## Behaviour

- Auto-balloon jobs remain FIFO across different users.
- A new scan can supersede unfinished work owned by the same user, preserving
  the existing reload/retry behaviour.
- A user cannot poll or cancel another user's job. An unknown job and a job
  owned by somebody else both return HTTP 404.
- The verified capacity remains one concurrent auto-balloon scan because the
  backend currently owns one `OcrPipeline` model bundle and one scan worker.
- Adding web-server threads does not increase the reported OCR capacity.

## Queue information returned with each job

- `queue_position`: one-based position among waiting jobs;
- `queue_depth`: total number of waiting jobs;
- `jobs_ahead`: running jobs plus earlier waiting jobs;
- `queue_wait_seconds`: time this job has spent waiting;
- `estimated_queue_wait_seconds`: capacity-aware estimate based on recent
  successful scan durations;
- `average_queue_wait_seconds`: recent average time from submission to start;
- `average_scan_duration_seconds`: recent average successful scan duration;
- sample counts for both averages.

The frontend displays queue position, current waiting time, estimated wait and
the recent average wait when enough samples exist.

## Lifetime and remaining scope

Queue state and the latest 20 timing samples are held in backend memory. They
reset when the backend restarts, which is appropriate while QTracX runs as one
backend process. A later deployment milestone must prevent multiple independent
web-server processes from each creating an uncoordinated model and queue.

Manual region OCR, title-field reads, reference detection and OCR-assisted PDF
text recovery do not enter the auto-balloon job queue, but their Paddle calls
now share its process-wide model-capacity gate. Interactive calls are admitted
before a waiting background call whenever the single model slot is released.
This keeps the existing FIFO auto-balloon queue intact while preventing direct
requests from running the same Paddle bundle concurrently.
