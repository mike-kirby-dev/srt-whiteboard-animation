---
name: srt-whiteboard-animation
description: Turns SRT subtitles into a whiteboard hand-drawing animation on warm cream paper: read the subtitles → output an illustration strategy → after confirmation, generate consistent-style line art → annotate mask regions by narrative meaning → adjust in the preview studio → render the MP4. Choreography keeps the mask-region reveal (annotation.json / sequence / startMs / protectedRegions), but the ink inside each region is replaced by stream's continuous pen strokes (skeleton/grid ink→color). Trigger when the user provides SRT subtitles and asks to "turn subtitles into a whiteboard hand-drawn / flowing pen-stroke video", "generate a whiteboard animation from SRT", or "storyboard and hand-draw by subtitles".
---

# SRT Whiteboard Animation (mask choreography + stream drawing)

Turns SRT subtitles into a whiteboard hand-drawing animation. **Choreography** keeps the mask-region reveal (regions are revealed in narrative order, regions that have not started stay completely hidden, and overlaps are guarded with `protectedRegions`); **drawing** switches to flowing pen strokes — within its own allowed mask, the pen tip glides continuously along the skeleton/grid laying down ink (ink pass → colour pass), and all regions share one persistent canvas so finished regions stay drawn. All user-facing explanations, storyboards, configuration and interface text must be in Chinese.

Unlike frame-by-frame jumps or rectangular wipes, this skill's pen strokes **flow continuously**; unlike whole-image stream, this skill draws region by region in **subtitle-narrative order**, so you can control each element's entrance order and timing.

## Default implementation parameters

| Item | Default requirement |
|---|---|
| Paper background | Generated images use warm cream aged-paper colour (`#F5EBD7` recommended); at render time the paper tone is sampled from the original image, inset from the four corners, and pure white is forbidden. |
| Drawing method | Continuous stream pen strokes per region: ink pass `ink` (lay down line art) → colour pass `color` (restore original colours); weight `ink:color = 2:1`. |
| Pen-stroke path | `--ink-path grid` (grid, default, stable) or `skeleton` (skeleton tracing, a closer fit for illustrations with clean line art). |
| Colouring style | `--color-fill contour-wipe` (contour sweep, default) or `brush` (brush along the trajectory). |
| Undrawn regions | A region's allowed mask = the rectangular `region` minus "later regions + protectedRegions"; regions that have not started are completely hidden. |
| Duration source | Each image's `sceneDurationMs` comes from the time span of that scene's subtitles (25–35 seconds per scene recommended). |
| Edit boxes | The preview studio shows all numbered edit boxes by default; edit boxes are not part of the animated picture. |

## Unified image-generation visual guidelines (mandatory)

Source images for all scenes must follow the same visual language; write the following requirements in full into the image-generation prompt before generating, and check them one by one afterwards:

- **Style and composition:** Minimal hand-drawn illustration, pure sketch style, a restrained Notion-like doodle aesthetic. Favour conceptual expression over realism; keep the composition simple with a clean background and plenty of white space, with a calm, clear overall mood, and consistent lines/characters/colours across the series.
- **Colour and material:** Beige paper background `#F5EBD7` and dark grey sketch lines; only red, orange and blue may be used, sparingly, as conceptual accents. No other accent colours, highly saturated palettes or complex textures.
- **Characters and objects:** Express objects with simple outlines, a few lines and white space, emphasising relationships/change/core concepts rather than true proportions, materials and detail.
- **Absolutely forbidden:** Any text, words, letters, numbers, fonts or labels in the scene source image; realism, photographic detail, 3D effects, painterly texture; complex scenes, dense backgrounds, elaborate decoration and highly saturated images.
- **Drawing-hand exception:** If the user explicitly states that the text on the pen barrel is their logo and asks to keep it, the logo on the `drawing-hand.png` pen barrel may stay; it does not count as scene source-image text and need not be removed or redrawn. Without the user's explicit statement, treat the picture as text-free.

## Confirmation gates (mandatory)

In the default workflow, **after every step you must stop and wait for the user's explicit confirmation** before starting the next. Before confirmation, do not generate the next step's images, annotations, previews, videos or merged files; do not treat "no reply", "an earlier blanket authorisation" or "the user didn't object" as confirmation. When the user asks to change the previous step, redo only that step, then wait for confirmation again once it is done.

The only automatic follow-on action is: **as soon as the annotation JSON is created, automatically open the preview studio and load the directory containing that JSON**; this is part of the step 3 deliverable and does not need a separate confirmation to "open the preview studio". If the browser's File System Access API requires a user gesture, use the browser interface to select this already-determined directory; do not ask the user for extra confirmation or tell them to open the preview studio themselves.

## Workflow

1. **Read the subtitles and produce a strategy (no image generation).** Use `scripts/parse_srt.py` to parse the SRT into subtitle entries and suggest a storyboard at 25–35 seconds per scene. From that, output the illustration strategy: for each scene, its scene number, core expression, main subject, corresponding subtitle range and `sceneDurationMs`. Each scene expresses only one core idea. **When done, stop and wait for the user to confirm the strategy.**
2. **Generate the line art.** Only after the user confirms the strategy, generate 16:9 warm cream aged-paper line-art images scene by scene according to the "Unified image-generation visual guidelines", with background `#F5EBD7` and ample white space between subjects to make automatic splitting easy; do not generate text, complex photos, overlapping objects or anything that conflicts with the guidelines. **When done, stop, show the line art and wait for the user to confirm.**
3. **Read the subtitles first, then look at the image, then annotate and open the preview studio.** Only after the user confirms the line art, first read the subtitles for that image, then actually view the image, and get the original image's pixel width and height; do not guess the picture from the subtitles alone, and do not order mechanically by position on screen alone. First distil the narrative events from the subtitles, then map the visible subjects in the image to those events, and arrange the drawing order semantically as "scene setup → key characters/objects → action, conflict or change → reaction/result". Then create `<image-name>.annotation.json`. Once created, immediately open `assets/preview.html` in the default browser and, through the preview studio's "Open folder", load **the directory containing that annotation file** with all its `<name>.png` + `<name>.annotation.json` pairs; do not merely give file paths or ask the user to do it themselves. **Once the preview studio has the directory loaded, stop and wait for the user to confirm the annotations and preview content.**
4. **Generate the region preview image.** Only after the user confirms the annotations and preview content, use `render_annotation_preview.py` to produce a numbered/direction check image, and verify that the regions match the narrative order, all regions are inside the canvas, and overlapping subjects are guarded with `protectedRegions`. **When done, stop and wait for the user to confirm the preview image.**
5. **Adjust and save in the preview studio.** Only after the user confirms the preview image, adjust in the already-open preview studio with the matching directory loaded: by default (not playing) it shows the full image and region boxes; the canvas is a **rectangle proxy**: drag a region's four edges and corners to change `region`; on the right, edit name/direction/**start (ms)/end (ms)** (duration = end − start, read-only) and **subtitle**; drag the module list to **reorder** (`sequence` is renumbered automatically); selecting a module automatically highlights its subtitle; drag the timeline or press play to watch the reveal (regions that have not started are not shown); `direction` only affects this proxy. When done, click "Save this scene"/"Save all" to write back to the original `.annotation.json` (including each region's `subtitle`, and aligning `sceneDurationMs` to the end of the last region + 0.5s). **After saving, stop and wait for the user to confirm the final annotations and timing.**
6. **Render the video from the command line.** Only after the user confirms the final annotations and timing, use `render_stream_whiteboard.py` to produce a fully-cleared MP4 scene by scene, and spot-check three time points: the opening, the middle of any overlapping module, and the ending. **When done, stop and wait for the user to confirm the finished video.**
7. **Merge scenes (multi-scene only).** Only after the user confirms all single-scene videos, use `merge_scenes.py` to merge them into one in order. **When done, stop and wait for the user to confirm the final merged video.**

## Directory convention

Create in the user's project:

```text
assets/whiteboard/<project-name>/
  scene-01-<name>.png
  scene-01-<name>.annotation.json     # same name as the png
  scene-01-<name>-whiteboard.mp4      # finished video
  scene-01-<name>-preview.mp4         # real clip (generated by the preview studio, low resolution)
```

The image and its config must share a name: `foo.png` pairs with `foo.annotation.json`. The preview studio uses this to load configs automatically.

## Semantic ordering and pixel-level annotation (must be done)

1. **Reading basis:** Before annotating you must have both the subtitles and the viewed original image. If either is missing, request it first; do not generate annotations.
2. **Ordering basis:** `sequence`, `startMs` and `label` must reflect the order of events in the subtitles, not merely left-to-right, top-to-bottom or visual prominence.
3. **Coordinate basis:** Each module outputs integer pixel `x`, `y`, `width`, `height` in the original image's coordinate system; the origin is the top-left corner, and percentages/ratios/estimated coordinates or omitted dimensions are forbidden. `canvas.width`/`canvas.height` must equal the original image's pixel dimensions.
4. **Module fields:** Each element contains `sequence`, `narrativeRole`, `subtitle`, `region`, `reveal`, `handPath`. `narrativeRole` explains, in Chinese, its narrative role in the subtitles; `subtitle` stores the subtitle text for that region (from the SRT, for preview-studio linkage and later use); `sequence` is continuous starting at 1.
5. **Validation:** Before generating the preview, check that each region is inside the canvas, covers the corresponding visible subject, and matches the subtitle events; for overlapping subjects, guard them with `protectedRegions` and then draw the module.

## Timing model (specific to stream drawing)

- **Total duration per scene** `sceneDurationMs` comes from that scene's subtitle time span (`scenes[].sceneDurationMs` from `parse_srt.py`).
- **Regions are drawn serially:** Stream drawing is one pen moving, so within one scene the regions should proceed **one after another in time** (`startMs` values do not overlap): the next region starts from the previous region's `startMs + durationMs` (plus an optional 100–300ms breath). If `startMs` values overlap, the renderer still processes them in order, but visually it is no longer concurrent.
- **ink→color inside a region:** Each region's `durationMs` is split into an ink-pass segment and a colour-pass segment at `ink:color = 2:1`. `durationMs` is determined by the preview studio's **start/end times** (end − start), and can be aligned to the duration of that region's subtitle; as an initial estimate you can also use 150 pixels/second × drawing distance.
- **Gaze hold at the end:** After all regions are drawn, the scene is automatically padded to `sceneDurationMs`, ensuring the complete original image is held for at least 0.5 seconds at the end.
- Under stream drawing, `reveal.direction` **does not determine the real pen strokes** (the strokes are generated automatically from the skeleton/grid); it only feeds the preview studio's rectangle-proxy demo. It is kept so the preview studio remains usable.

## Mask invariants (choreography layer, must be enforced)

- At time `t`, a module may show only pixels after its `reveal.startMs ≤ t` and not beyond the current drawing progress; no lines/fills/images of a module that has not started may appear.
- Each region's **allowed mask** = the rectangular `region` minus the `region` of all **later modules**, then minus this module's `reveal.protectedRegions`. Stream inking is confined to the allowed mask, so later regions never leak lines early.
- `protectedRegions` use the same original-image integer pixel coordinates as `region`, for cases where the rectangle is too large, subjects overlap, or background lines might leak.
- The renderer already implements the order "ink confined to the allowed mask → later regions and protected regions are naturally untouched"; the preview studio's rectangle proxy demonstrates the same choreography with an equivalent `destination-out` subtraction.

## Config example

```json
{
  "sceneId": "scene-01",
  "canvas": { "width": 1672, "height": 941 },
  "storyBasis": "Summary of the events in this scene's subtitles",
  "sceneDurationMs": 9000,
  "elements": [
    {
      "id": "rockery",
      "label": "Rockery scene",
      "sequence": 1,
      "narrativeRole": "Scene setup for the story",
      "subtitle": "On Monkey Mountain, a little monkey sits on top of the rockery, holding a banana.",
      "type": "structure",
      "region": { "x": 20, "y": 120, "width": 540, "height": 780 },
      "reveal": { "direction": "top_to_bottom", "startMs": 300, "durationMs": 2600, "maskPaddingPx": 22, "protectedRegions": [] },
      "handPath": { "start": [290, 130], "end": [290, 890], "easing": "easeInOut" }
    }
  ]
}
```

> `direction` / `handPath` are used only by the preview studio's rectangle proxy; the finished video's pen strokes are generated automatically by stream and need no fine-tuning.

## Using the scripts

Run all render scripts with the interpreter from the skill's own `.venv` (isolated dependencies).

1. **Prepare the environment** (first time, or when dependencies are missing):
   ```bash
   python scripts/prepare_env.py --check   # probe; on success the last line prints ENV_PY=<path>, capture it for later
   python scripts/prepare_env.py           # if missing, creates .venv and installs opencv-python/numpy/av
   ```
2. **Parse subtitles + suggest a storyboard**:
   ```bash
   python scripts/parse_srt.py <subtitles.srt> --target-sec 30 --min-sec 25 --max-sec 35
   ```
3. **Numbered region preview image**:
   ```bash
   python scripts/render_annotation_preview.py <image> <annotation> <preview-output>
   ```
4. **Preview studio (no server needed)**: open `assets/preview.html` directly in Chrome / Edge, click "Open folder" and pick the directory → all images + same-named annotations are loaded → drag to edit → "Save" writes back to the original files. Writing back needs the File System Access API (Chrome/Edge); other browsers download the file instead and you overwrite manually. Rendering still goes through the command line (step 5 below).
5. **Render a single-scene video**:
   ```bash
   <ENV_PY> scripts/render_stream_whiteboard.py <image> <annotation> <output.mp4> assets/drawing-hand.png \
       [--ink-path grid|skeleton] [--color-fill contour-wipe|brush] [--total-ms <milliseconds>]
   ```
   When `--total-ms` is omitted, the annotation's `sceneDurationMs` is used. The last line prints `OUTPUT=<path>`.
6. **Merge scenes**:
   ```bash
   <ENV_PY> scripts/merge_scenes.py --inputs scene1.mp4 scene2.mp4 scene3.mp4 --output final.mp4
   ```

## Quality checks

Confirm before/after rendering:

- The first frame is a clean warm cream aged-paper background, with no lines revealed early.
- The corresponding subtitles have been read and the original image actually viewed; `canvas` matches the original image's pixel size, and every `region` is integer pixel coordinates inside the canvas.
- `sequence` and `startMs` match the order of the subtitle events; the preview image's numbers/labels/regions come from the same annotation JSON.
- Check three time points — the opening, the middle of any overlapping module, and after all modules are done: undrawn modules are all invisible, protected overlap regions do not leak, and the final frame shows the complete original image.
- The pen tip stays close to the advancing pen strokes; for illustrations with clean line art, use `--ink-path skeleton` to make the strokes fit more closely.
- After all modules finish, hold the complete original image for at least 0.5 seconds.
- After merging scenes, the order and duration match the subtitle storyboard.

To change the result, first adjust the annotations (regions/order/timing) in the preview studio (`assets/preview.html`) and save, then render from the command line; do not re-render blindly over and over.
