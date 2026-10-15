# Reference video context

## V4 / V8 from Windows or Linux

Pass the video at startup; no source edit is necessary. `start_mac_vlm.py`,
`start_notebook_test.py` and `start_drawer_test.py` all accept:

```powershell
python wam_direct/start_notebook_test.py --reference-video "/home/liberai/demos/notebook.mp4" --reference-video-frames 12
```

The path is an existing **Leo3 absolute path**, even when this command runs on
Windows. Copy a Windows video to Leo3 first. Paths with spaces must be quoted.
The examples are placeholders; choose the demonstration for the actual task.
The same arguments work with `start_mac_vlm.py --control-version 8` and its
normal task/model/dynamics options. The startup script prints whether a video
was configured and forwards its path and frame count into `leo3_command.sh`.
The runner checks/decodes it before creating the robot; a bad video fails startup
instead of silently omitting the demonstration.

Every takeover and subsequent decision request contains the same demonstration
frames, including V8 candidate review and new L2 requests. In the model's image
list, live observations and WAM history come first, V8 synthetic predictions
remain labeled separately, and demonstration frames come last with frame indices
and timestamps. The supervisor log records `REFERENCE VIDEO request=... frames=...`.
The shared image path also applies to the Opus/Gemini/GLM supervisor choices.

This is **sampled video context**, not native full-video playback: default 12
chronological frames, configurable from 1 to 32. Increase sampling when important
contact stages are missed. Demonstrated motion is a reference for task intent,
not proof of the live scene or an action sequence to replay.

These options apply to a **new session**; editing code does not add a video to an
already running session. Omitting `--reference-video` preserves no-demo behavior.

On top of baseline d192e04, the runner accepts these optional arguments for both
`--supervisor-backend local` and `--supervisor-backend mac`:

```bash
--reference-video "/home/liberai/下载/V16_collect_1783576048_5.mp4" \
--reference-video-frames 12
```

The video must exist on the runner machine (leo3). OpenCV is required there.
Startup samples 1–32 chronological frames, default 12, before robot creation.
Frames are JPEGs with a maximum side of 960 pixels. Each request includes source
checksum, frame indices and timestamps. Mac supervision transports and checks
these same images; no separate Mac video path is needed.

The demonstration is visual task context for Astra, separate from live camera
observations and WAM history. It is not sent to the WAM server. Uniform samples
are not L2 annotations. No index JSON, budget or control-policy changes are added.
Without these arguments the baseline behavior is unchanged. Inspect
`reference_video.json` and `reference_video/` inside the new runner session to
verify extraction. Restart with a new session to change the video.
