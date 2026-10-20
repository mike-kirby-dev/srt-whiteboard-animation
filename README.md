# SRT Whiteboard Animation Skill

> English translation of [geeklee/srt-whiteboard-animation](https://github.com/geeklee/srt-whiteboard-animation) (MIT).

A Skill that turns SRT subtitles into a whiteboard hand-drawing video, drawn in narrative order. It combines **mask-region choreography** with **flowing pen-stroke drawing**: each element appears in step with the subtitles, the pen tip lays down ink continuously within its region, then colour is added gradually, and the result is exported as an MP4.

It suits knowledge explainers, story narration, course subtitles or short-video scripts that you want as hand-drawn animation on warm cream paper.

## Example

**Scene: Monkeys fighting over a banana on Monkey Mountain** — following the narrative order of the subtitles, it draws the rockery and the little monkey, the big monkey snatching the banana, and the children looking on.

![Monkeys fighting over a banana on Monkey Mountain: SRT whiteboard animation demo](examples/scene-01-monkey-mountain-stream.gif)

Original line art: [view PNG](examples/scene-01-monkey-mountain.png).

## Core capabilities

- Parses SRT subtitles and splits them into scenes of the suggested 25–35 seconds each
- Outputs a storyboard and illustration strategy first, so each scene expresses just one core idea
- Builds a semantic drawing order for elements based on subtitle events, not on screen coordinates
- Manages regions, timing, subtitle links and overlap-protected regions with `annotation.json`
- Uses continuous flowing pen strokes in every region: `ink` lays down the line art first, then `color` adds colour
- Supports a browser preview studio for adjusting regions, order, timing and subtitle links
- Supports scene-by-scene rendering and multi-scene merging, outputting a complete MP4

## How it works

The key to this Skill is "subtitle-driven, confirmed step by step". After each step it waits for confirmation, so rendering effort is not wasted while the storyboard, line art or annotations are still unsettled:

1. Parse the SRT and output the storyboard and illustration strategy.
2. After confirmation, generate line art in a consistent style.
3. After the line art is confirmed, create annotations from the subtitles and the original image, and load them into the preview studio.
4. After the annotations are confirmed, generate region and direction check images.
5. In the preview studio, adjust regions, narrative order, timing and subtitle links, then save.
6. After the final annotations are confirmed, render the MP4 scene by scene.
7. For multi-scene projects, merge the scenes after each finished scene is confirmed.

## Visual guidelines

- Warm cream paper background: `#F5EBD7` recommended
- Dark grey sketch lines, with red, orange and blue used only sparingly as conceptual accents
- Minimal hand-drawn style, clean background and plenty of white space
- No in-scene text, labels, photographic look, 3D effects or complex textures

## Installation and environment

The Skill ships its own script for preparing an isolated Python virtual environment. Run this on first use:

```bash
python scripts/prepare_env.py --check
python scripts/prepare_env.py
```

On success, the first command prints `ENV_PY=<path>`; use that interpreter for later renders to keep dependencies isolated.

## Project asset structure

```text
assets/whiteboard/<project-name>/
├── scene-01-<name>.png
├── scene-01-<name>.annotation.json
├── scene-01-<name>-whiteboard.mp4
└── scene-01-<name>-preview.mp4
```

The image and its annotation must share a name, e.g. `scene-01-demo.png` pairs with `scene-01-demo.annotation.json`.

## Annotation format

Each element uses integer pixel coordinates of the original image and is linked to events in the subtitles through `sequence`, `subtitle` and `narrativeRole`. Order regions as "scene setup → key characters/objects → action or change → reaction/result".

```json
{
  "sceneId": "scene-01",
  "canvas": { "width": 1672, "height": 941 },
  "storyBasis": "The little monkey holds a banana on Monkey Mountain, the big monkey snatches it, and the children watch.",
  "sceneDurationMs": 9000,
  "elements": [
    {
      "id": "rockery",
      "label": "Monkey Mountain scene",
      "sequence": 1,
      "narrativeRole": "Scene setup for the story",
      "subtitle": "The little monkey sits on top of Monkey Mountain, holding a banana.",
      "type": "structure",
      "region": { "x": 20, "y": 120, "width": 540, "height": 780 },
      "reveal": {
        "direction": "top_to_bottom",
        "startMs": 300,
        "durationMs": 2600,
        "maskPaddingPx": 22,
        "protectedRegions": []
      },
      "handPath": { "start": [290, 130], "end": [290, 890], "easing": "easeInOut" }
    }
  ]
}
```

`direction` and `handPath` drive the preview studio's rectangle proxy; the real pen strokes in the final video are generated automatically by the stream renderer. For objects that overlap, mark the areas that must appear later in the earlier element's `protectedRegions`, so later content is not revealed too early.

## Common commands

Parse subtitles and generate a suggested storyboard:

```bash
python scripts/parse_srt.py <subtitles.srt> --target-sec 30 --min-sec 25 --max-sec 35
```

Generate a region check image:

```bash
python scripts/render_annotation_preview.py <image-path> <annotation-path> <preview-output-path>
```

Open `assets/preview.html` and use "Open folder" to load the scene directory, where you can edit regions, order, timing and subtitle links.

Render a single scene:

```bash
<ENV_PY> scripts/render_stream_whiteboard.py <image-path> <annotation-path> <output.mp4> assets/drawing-hand.png \
  --ink-path grid --color-fill contour-wipe
```

Merge multiple scenes:

```bash
<ENV_PY> scripts/merge_scenes.py --inputs scene1.mp4 scene2.mp4 scene3.mp4 --output final.mp4
```

## Quality checks

- The first frame is clean warm cream paper, with no lines revealed early
- `canvas` matches the original image size, and all regions are integer pixel coordinates inside the canvas
- `sequence` and `startMs` match the narrative order of the subtitles
- In mid-scene frames, regions that have not started and protected regions do not appear early
- The pen tip stays close to the current pen strokes; when the line art is clear, you can choose `--ink-path skeleton`
- Each scene holds the complete picture for at least 0.5 seconds at the end; the merged scene order matches the subtitle storyboard

## Repository contents

```text
srt-whiteboard-animation/
├── SKILL.md                         # Complete workflow and constraints
├── assets/
│   ├── drawing-hand.png              # Hand asset
│   ├── preview.html                  # Local editing preview studio
├── examples/                         # README example assets
├── scripts/
│   ├── parse_srt.py                  # Subtitle parsing and storyboard suggestions
│   ├── render_annotation_preview.py  # Annotation check images
│   ├── render_stream_whiteboard.py   # Flowing pen-stroke MP4 renderer
│   ├── merge_scenes.py               # Multi-scene merging
│   └── prepare_env.py                # Dependency environment setup
└── agents/openai.yaml                # Codex metadata
```

## Contributing

Issues and Pull Requests are welcome. Any change that touches the drawing logic should be checked against real subtitles, annotations and finished videos, to verify mask protection, timing and the final picture.

## License

This project is open source under the MIT License; see [LICENSE](LICENSE).

## About the author

A fish-keeping old-timer / AI Builder / building a one-person company with an AI team.

Original author on Douyin, Bilibili and WeChat Official Account: 江哥是老登啊 (Jiang Ge Shi Lao Deng A)
