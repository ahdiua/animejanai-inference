#!/usr/bin/env python3
"""Real FFmpeg regression coverage for aji_gop_media."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def run(*args, check=True):
    return subprocess.run(args, check=check, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True)


def make_fixture(path, codec, *, frames=97, open_gop=False, vfr=False):
    if codec == "h264":
        encoder = "libx264"
        params = "keyint=24:min-keyint=24:scenecut=0:open-gop=" + ("1" if open_gop else "0")
        codec_args = ["-c:v", encoder, "-preset", "ultrafast", "-bf", "2",
                      "-x264-params", params]
    else:
        encoder = "libx265"
        params = "keyint=24:min-keyint=24:scenecut=0:open-gop=" + ("1" if open_gop else "0")
        codec_args = ["-c:v", encoder, "-preset", "ultrafast", "-bf", "2",
                      "-x265-params", params]

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=24000/1001",
        "-frames:v", str(frames),
    ]
    if vfr:
        command += ["-vf", "setpts=PTS+gte(N\\,30)*0.5/TB", "-fps_mode", "passthrough"]
    command += codec_args + ["-pix_fmt", "yuv420p", str(path)]
    run(*command)


def scan(tool, path, *, success=True):
    result = run(str(tool), "scan", str(path), check=False)
    if success:
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    assert result.returncode != 0
    assert result.stderr.strip()
    assert result.stdout == ""
    return None


def frame_hashes(path):
    result = run("ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
                 "-map", "0:v:0", "-f", "framemd5", "-")
    hashes = []
    for line in result.stdout.splitlines():
        if line and not line.startswith("#"):
            hashes.append(line.rsplit(",", 1)[-1].strip())
    return hashes


def probe_video(path):
    result = run("ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name,width,height,pix_fmt",
                 "-of", "json", str(path))
    return json.loads(result.stdout)["streams"][0]


def assert_index(index, codec):
    required = {
        "version", "video_stream", "codec", "width", "height", "pix_fmt",
        "time_base", "fps", "start_pts", "frames", "packets", "gops",
    }
    assert set(index) == required
    assert index["version"] == 1
    assert index["video_stream"] == 0
    assert index["codec"] == codec
    assert index["width"] == 64 and index["height"] == 48
    assert index["pix_fmt"] == "yuv420p"
    assert index["time_base"] == [1, 1000]
    assert index["fps"] == [24000, 1001]
    assert index["start_pts"] == 0
    assert index["frames"] == 97
    assert index["packets"] == 97
    assert sum(gop["frames"] for gop in index["gops"]) == 97
    assert [gop["frames"] for gop in index["gops"]] == [24, 24, 24, 24, 1]
    assert [gop["frame"] for gop in index["gops"]] == [0, 24, 48, 72, 96]
    assert [gop["packet"] for gop in index["gops"]] == [0, 24, 48, 72, 96]
    assert [gop["pts"] for gop in index["gops"]] == [0, 1001, 2002, 3003, 4004]
    assert all(set(gop) == {"packet", "frame", "pts", "frames"}
               for gop in index["gops"])


def exercise_closed_fixture(tool, root, codec):
    source = root / f"closed {codec} 动画.mkv"
    pieces = root / f"pieces {codec} 分片"
    invalid = root / f"invalid {codec}"
    no_cut = root / f"no cut {codec}"
    subset = root / f"subset {codec}"
    source.parent.mkdir(parents=True, exist_ok=True)
    pieces.mkdir()
    invalid.mkdir()
    no_cut.mkdir()
    subset.mkdir()
    make_fixture(source, codec)

    index = scan(tool, source)
    assert_index(index, codec)
    cuts = [str(gop["packet"]) for gop in index["gops"][1:]]
    result = run(str(tool), "split", str(source), str(pieces), *cuts, check=False)
    assert result.returncode == 0, result.stderr

    outputs = sorted(pieces.glob("gop-*.mkv"))
    assert len(outputs) == 5
    combined = []
    source_meta = probe_video(source)
    for output in outputs:
        assert probe_video(output) == source_meta
        combined.extend(frame_hashes(output))
    assert combined == frame_hashes(source)

    result = run(str(tool), "split", str(source), str(no_cut), check=False)
    assert result.returncode == 0, result.stderr
    no_cut_outputs = sorted(no_cut.glob("gop-*.mkv"))
    assert [path.name for path in no_cut_outputs] == ["gop-000000.mkv"]
    assert frame_hashes(no_cut_outputs[0]) == frame_hashes(source)

    result = run(str(tool), "split", str(source), str(subset),
                 str(index["gops"][2]["packet"]), check=False)
    assert result.returncode == 0, result.stderr
    subset_outputs = sorted(subset.glob("gop-*.mkv"))
    assert [len(frame_hashes(path)) for path in subset_outputs] == [48, 49]
    assert [item for path in subset_outputs for item in frame_hashes(path)] == frame_hashes(source)

    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in outputs}
    result = run(str(tool), "split", str(source), str(pieces), *cuts, check=False)
    assert result.returncode != 0
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in outputs}
    assert after == before

    result = run(str(tool), "split", str(source), str(invalid), "1", check=False)
    assert result.returncode != 0
    assert list(invalid.iterdir()) == []

    if codec == "h264":
        with open("/dev/full", "w", encoding="utf-8") as full:
            result = subprocess.run([str(tool), "scan", str(source)], stdout=full,
                                    stderr=subprocess.PIPE, text=True)
        assert result.returncode != 0
        assert "stdout" in result.stderr


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: sanity/gop_media.py PATH_TO_AJI_GOP_MEDIA")
    tool = Path(sys.argv[1]).resolve()
    if not tool.is_file():
        raise AssertionError(f"missing executable: {tool}")

    with tempfile.TemporaryDirectory(prefix="aji gop 测试 ") as temporary:
        root = Path(temporary)
        exercise_closed_fixture(tool, root, "h264")
        exercise_closed_fixture(tool, root, "hevc")

        vfr = root / "vfr.mkv"
        make_fixture(vfr, "h264", vfr=True)
        scan(tool, vfr, success=False)

        open_gop = root / "open-gop.mkv"
        make_fixture(open_gop, "h264", open_gop=True)
        scan(tool, open_gop, success=False)


if __name__ == "__main__":
    main()
