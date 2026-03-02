"""
Generate a looping GIF animation for LinkedIn promoting a 9-part Redis blog series.
1080x1080, Redis-themed colors, optimized for LinkedIn.
"""

from PIL import Image, ImageDraw, ImageFont
import math
import os

# --- Config ---
W, H = 1080, 1080
BG = (26, 26, 46)        # #1a1a2e
RED = (220, 56, 45)      # #DC382D
WHITE = (255, 255, 255)
RED_GLOW = (220, 56, 45, 80)
FPS = 20  # frames per second for smooth animation

PARTS = [
    "Part 0: Foundation",
    "Part 1: Architecture & Event Loop",
    "Part 2: Data Structures Deep Dive",
    "Part 3: Memory & Persistence",
    "Part 4: Networking & Performance",
    "Part 5: Replication & Sentinel",
    "Part 6: Redis Cluster",
    "Part 7: Use Cases & Patterns",
    "Part 8: Production War Stories",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "redis_deep_dive_linkedin.gif")


def get_font(size, bold=False):
    """Try to load a good font, fall back to default."""
    font_candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    if bold:
        font_candidates = [
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
    for f in font_candidates:
        if os.path.exists(f):
            try:
                return ImageFont.truetype(f, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_redis_logo(draw, cx, cy, scale=1.0, alpha=1.0):
    """Draw a stylized Redis diamond/cube logo shape."""
    s = 80 * scale
    # Outer diamond
    points = [(cx, cy - s * 1.2), (cx + s * 1.5, cy), (cx, cy + s * 1.2), (cx - s * 1.5, cy)]
    # Glow layers
    if alpha > 0.3:
        for glow_i in range(3, 0, -1):
            glow_s = s + glow_i * 12
            glow_pts = [
                (cx, cy - glow_s * 1.2),
                (cx + glow_s * 1.5, cy),
                (cx, cy + glow_s * 1.2),
                (cx - glow_s * 1.5, cy),
            ]
            glow_alpha = int(30 * alpha * (1 - glow_i / 4))
            glow_color = (220, 56, 45, glow_alpha)
            # Draw on a temp overlay for glow
            overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            od.polygon(glow_pts, fill=glow_color)
            # Composite onto main - we'll handle this in the caller

    # Main diamond fill
    fill_r = int(220 * alpha)
    fill_g = int(56 * alpha)
    fill_b = int(45 * alpha)
    draw.polygon(points, fill=(fill_r, fill_g, fill_b))

    # Inner lines for "stacked" look
    mid_y = cy - s * 0.3
    draw.line([(cx - s * 1.3, cy), (cx, mid_y), (cx + s * 1.3, cy)],
              fill=(255, 255, 255, int(200 * alpha)) if alpha < 1 else (255, 200, 200), width=3)
    draw.line([(cx - s * 1.1, cy + s * 0.35), (cx, cy + s * 0.05), (cx + s * 1.1, cy + s * 0.35)],
              fill=(255, 255, 255, int(150 * alpha)) if alpha < 1 else (255, 180, 180), width=2)


def draw_redis_logo_rgba(base_img, cx, cy, scale=1.0, glow_intensity=1.0):
    """Draw Redis logo with proper glow using RGBA compositing."""
    s = 80 * scale

    # Glow layers
    for glow_i in range(5, 0, -1):
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        glow_s = s + glow_i * 15
        glow_pts = [
            (cx, cy - glow_s * 1.2),
            (cx + glow_s * 1.5, cy),
            (cx, cy + glow_s * 1.2),
            (cx - glow_s * 1.5, cy),
        ]
        glow_alpha = int(40 * glow_intensity * (1 - glow_i / 6))
        od.polygon(glow_pts, fill=(220, 56, 45, glow_alpha))
        base_img = Image.alpha_composite(base_img, overlay)

    # Main diamond
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    points = [(cx, cy - s * 1.2), (cx + s * 1.5, cy), (cx, cy + s * 1.2), (cx - s * 1.5, cy)]
    od.polygon(points, fill=(220, 56, 45, 255))

    # Inner detail lines
    mid_y = cy - s * 0.3
    od.line([(cx - s * 1.3, cy), (cx, mid_y), (cx + s * 1.3, cy)], fill=(255, 200, 200, 200), width=3)
    od.line([(cx - s * 1.1, cy + s * 0.35), (cx, cy + s * 0.05), (cx + s * 1.1, cy + s * 0.35)],
            fill=(255, 180, 180, 150), width=2)

    base_img = Image.alpha_composite(base_img, overlay)
    return base_img


def make_base():
    """Create base dark background with subtle gradient."""
    img = Image.new("RGBA", (W, H), BG + (255,))
    draw = ImageDraw.Draw(img)
    # Subtle radial-ish gradient overlay
    for y in range(H):
        for x in range(0, W, 4):  # step 4 for speed
            dist = math.sqrt((x - W / 2) ** 2 + (y - H / 2) ** 2)
            max_dist = math.sqrt((W / 2) ** 2 + (H / 2) ** 2)
            factor = 1 - (dist / max_dist) * 0.3
            r = int(BG[0] * factor)
            g = int(BG[1] * factor)
            b = int(BG[2] * factor)
            draw.rectangle([x, y, x + 3, y], fill=(r, g, b, 255))
    return img


def make_base_fast():
    """Create base dark background - simple solid for speed."""
    img = Image.new("RGBA", (W, H), BG + (255,))
    draw = ImageDraw.Draw(img)
    # Subtle vignette with concentric rectangles
    for i in range(20):
        inset = i * 27
        darkness = int(i * 3)
        color = (max(0, BG[0] - darkness), max(0, BG[1] - darkness), max(0, BG[2] - darkness), 8)
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rectangle([0, 0, W, inset], fill=color)
        od.rectangle([0, H - inset, W, H], fill=color)
        od.rectangle([0, 0, inset, H], fill=color)
        od.rectangle([W - inset, 0, W, H], fill=color)
        img = Image.alpha_composite(img, overlay)
    return img


def draw_decorative_lines(img):
    """Add subtle decorative elements."""
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # Top and bottom accent lines
    draw.rectangle([100, 50, W - 100, 53], fill=(220, 56, 45, 60))
    draw.rectangle([100, H - 53, W - 100, H - 50], fill=(220, 56, 45, 60))
    # Corner accents
    corner_len = 60
    corner_w = 3
    for x, y, dx, dy in [
        (80, 40, 1, 0), (80, 40, 0, 1),
        (W - 80, 40, -1, 0), (W - 80, 40, 0, 1),
        (80, H - 40, 1, 0), (80, H - 40, 0, -1),
        (W - 80, H - 40, -1, 0), (W - 80, H - 40, 0, -1),
    ]:
        x2 = x + dx * corner_len
        y2 = y + dy * corner_len
        draw.line([(x, y), (x2, y2)], fill=(220, 56, 45, 120), width=corner_w)
    return Image.alpha_composite(img, overlay)


def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


# ===================== FRAME GENERATORS =====================

def generate_frame1_frames(base_img):
    """Frame 1: Redis logo appears with glow effect (2 seconds)."""
    frames = []
    n_frames = FPS * 2  # 2 seconds

    font_label = get_font(28, bold=False)

    for i in range(n_frames):
        t = i / n_frames  # 0 -> 1
        img = base_img.copy()

        # Scale eases in
        if t < 0.5:
            scale = 0.5 + 0.5 * ease_out_back(t * 2)
            glow = t * 2
        else:
            scale = 1.0
            glow = 1.0 + 0.3 * math.sin((t - 0.5) * 4 * math.pi)  # pulsing glow

        img = draw_redis_logo_rgba(img, W // 2, H // 2 - 40, scale=scale, glow_intensity=glow)

        # "REDIS" text fades in during second half
        if t > 0.4:
            text_alpha = min(255, int(255 * (t - 0.4) / 0.3))
            overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            font_big = get_font(60, bold=True)
            tw, th = text_size(od, "REDIS", font_big)
            od.text(((W - tw) // 2, H // 2 + 100), "REDIS", font=font_big, fill=(255, 255, 255, text_alpha))

            # Small tagline
            if t > 0.6:
                tag_alpha = min(255, int(255 * (t - 0.6) / 0.3))
                tw2, th2 = text_size(od, "The Real-Time Data Platform", font_label)
                od.text(((W - tw2) // 2, H // 2 + 170), "The Real-Time Data Platform",
                        font=font_label, fill=(180, 180, 180, tag_alpha))

            img = Image.alpha_composite(img, overlay)

        img = draw_decorative_lines(img)
        frames.append(img.convert("RGB"))

    return frames


def generate_frame2_frames(base_img):
    """Frame 2: 'Redis Deep Dive' types in letter by letter (2 seconds)."""
    frames = []
    n_frames = FPS * 2
    title = "Redis Deep Dive"
    font_title = get_font(72, bold=True)
    font_sub = get_font(32, bold=False)
    subtitle = "A 9-Part Technical Series"

    for i in range(n_frames):
        t = i / n_frames
        img = base_img.copy()

        # Small logo at top
        img = draw_redis_logo_rgba(img, W // 2, 220, scale=0.5, glow_intensity=0.6)

        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)

        # Typing effect
        chars_to_show = int(len(title) * min(1.0, t / 0.7))
        shown_text = title[:chars_to_show]

        # Cursor blink
        cursor = "|" if (i // (FPS // 4)) % 2 == 0 and t < 0.8 else ""
        display = shown_text + cursor

        tw, th = text_size(od, title, font_title)
        # Draw shown text in red
        od.text(((W - tw) // 2, H // 2 - 20), shown_text, font=font_title, fill=RED + (255,))

        # Draw cursor
        if cursor:
            partial_w, _ = text_size(od, shown_text, font_title)
            cx = (W - tw) // 2 + partial_w
            od.text((cx, H // 2 - 20), "|", font=font_title, fill=(255, 255, 255, 200))

        # Subtitle fades in after typing
        if t > 0.75:
            sub_alpha = min(255, int(255 * (t - 0.75) / 0.2))
            stw, sth = text_size(od, subtitle, font_sub)
            od.text(((W - stw) // 2, H // 2 + 70), subtitle, font=font_sub, fill=(180, 180, 180, sub_alpha))

            # Decorative line under title
            line_w = int(tw * min(1.0, (t - 0.75) / 0.15))
            line_x = (W - tw) // 2
            od.rectangle([line_x, H // 2 + 55, line_x + line_w, H // 2 + 58], fill=RED + (sub_alpha,))

        img = Image.alpha_composite(img, overlay)
        img = draw_decorative_lines(img)
        frames.append(img.convert("RGB"))

    return frames


def generate_frame3_frames(base_img):
    """Frame 3: Parts scroll through (1 second each, 9 parts = 9 seconds)."""
    frames = []
    font_title = get_font(48, bold=True)
    font_part = get_font(38, bold=True)
    font_part_desc = get_font(30, bold=False)
    font_counter = get_font(22, bold=False)

    frames_per_part = FPS * 1  # 1 second per part

    for part_idx in range(len(PARTS)):
        for fi in range(frames_per_part):
            t = fi / frames_per_part  # 0->1 within this part

            img = base_img.copy()
            # Small logo at top
            img = draw_redis_logo_rgba(img, W // 2, 180, scale=0.35, glow_intensity=0.5)

            overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)

            # "Redis Deep Dive" title at top
            ttw, tth = text_size(od, "Redis Deep Dive", font_title)
            od.text(((W - ttw) // 2, 270), "Redis Deep Dive", font=font_title, fill=RED + (255,))

            # Decorative line
            od.rectangle([(W - ttw) // 2, 330, (W + ttw) // 2, 333], fill=RED + (120,))

            # Progress dots
            dot_y = 360
            total_dots_w = len(PARTS) * 24
            dot_start_x = (W - total_dots_w) // 2
            for di in range(len(PARTS)):
                dx = dot_start_x + di * 24
                if di < part_idx:
                    od.ellipse([dx, dot_y, dx + 12, dot_y + 12], fill=RED + (255,))
                elif di == part_idx:
                    pulse = int(4 * math.sin(t * math.pi * 2))
                    od.ellipse([dx - pulse, dot_y - pulse, dx + 12 + pulse, dot_y + 12 + pulse],
                               fill=RED + (255,))
                else:
                    od.ellipse([dx, dot_y, dx + 12, dot_y + 12], fill=(100, 100, 100, 150))

            # Current part - slide in from right
            part_text = PARTS[part_idx]
            # Split into part number and description
            colon_idx = part_text.index(":")
            part_num = part_text[:colon_idx]
            part_desc = part_text[colon_idx + 1:].strip()

            # Slide + fade animation
            if t < 0.2:
                # Slide in
                offset_x = int(200 * (1 - ease_out_cubic(t / 0.2)))
                alpha = int(255 * (t / 0.2))
            elif t > 0.85:
                # Slide out
                offset_x = int(-200 * ease_in_cubic((t - 0.85) / 0.15))
                alpha = int(255 * (1 - (t - 0.85) / 0.15))
            else:
                offset_x = 0
                alpha = 255

            # Part number (big, red)
            pnw, pnh = text_size(od, part_num, font_title)
            od.text(((W - pnw) // 2 + offset_x, 440), part_num, font=font_title,
                    fill=(220, 56, 45, alpha))

            # Part description
            pdw, pdh = text_size(od, part_desc, font_part)
            od.text(((W - pdw) // 2 + offset_x, 510), part_desc, font=font_part,
                    fill=(255, 255, 255, alpha))

            # Decorative box around current part
            box_alpha = int(60 * (alpha / 255))
            box_w = max(pnw, pdw) + 80
            box_x = (W - box_w) // 2 + offset_x
            od.rectangle([box_x, 425, box_x + box_w, 560], outline=(220, 56, 45, box_alpha), width=2)

            # Show dimmed list of nearby parts for context
            context_font = get_font(24, bold=False)
            for offset in [-2, -1, 1, 2]:
                neighbor_idx = part_idx + offset
                if 0 <= neighbor_idx < len(PARTS):
                    neighbor_alpha = int(60 * (alpha / 255))
                    ny = 490 + offset * 70
                    if 600 <= ny <= 800:
                        ntw, nth = text_size(od, PARTS[neighbor_idx], context_font)
                        od.text(((W - ntw) // 2, ny), PARTS[neighbor_idx],
                                font=context_font, fill=(150, 150, 150, neighbor_alpha))

            # Counter
            counter_text = f"{part_idx + 1} / {len(PARTS)}"
            ctw, cth = text_size(od, counter_text, font_counter)
            od.text(((W - ctw) // 2, H - 120), counter_text, font=font_counter,
                    fill=(150, 150, 150, 180))

            img = Image.alpha_composite(img, overlay)
            img = draw_decorative_lines(img)
            frames.append(img.convert("RGB"))

    return frames


def generate_frame4_frames(base_img):
    """Frame 4: 'Read the complete series' with link icon (3 seconds)."""
    frames = []
    n_frames = FPS * 3
    font_cta = get_font(52, bold=True)
    font_sub = get_font(28, bold=False)
    font_arrow = get_font(60, bold=True)

    cta_text = "Read the Complete Series"
    sub_text = "Link in comments"

    for i in range(n_frames):
        t = i / n_frames
        img = base_img.copy()

        # Logo
        glow = 0.6 + 0.3 * math.sin(t * math.pi * 2)
        img = draw_redis_logo_rgba(img, W // 2, 280, scale=0.5, glow_intensity=glow)

        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)

        # Title
        font_title = get_font(44, bold=True)
        ttw, _ = text_size(od, "Redis Deep Dive", font_title)
        od.text(((W - ttw) // 2, 380), "Redis Deep Dive", font=font_title, fill=RED + (255,))

        # CTA appears
        if t > 0.1:
            cta_alpha = min(255, int(255 * (t - 0.1) / 0.2))
            ctw, cth = text_size(od, cta_text, font_cta)

            # Draw button-like background
            btn_pad_x, btn_pad_y = 50, 20
            btn_x = (W - ctw) // 2 - btn_pad_x
            btn_y = 490 - btn_pad_y
            btn_w = ctw + btn_pad_x * 2
            btn_h = cth + btn_pad_y * 2

            # Button background
            od.rounded_rectangle(
                [btn_x, btn_y, btn_x + btn_w, btn_y + btn_h],
                radius=15,
                fill=(220, 56, 45, cta_alpha),
            )
            od.text(((W - ctw) // 2, 490), cta_text, font=font_cta, fill=(255, 255, 255, cta_alpha))

        # Link icon / arrow animation
        if t > 0.3:
            arrow_alpha = min(255, int(255 * (t - 0.3) / 0.2))
            bounce = int(8 * math.sin(t * math.pi * 3))
            # Draw a simple link chain icon using text
            arrow = "→"
            atw, ath = text_size(od, arrow, font_arrow)
            od.text(((W - atw) // 2, 570 + bounce), arrow, font=font_arrow, fill=(255, 255, 255, arrow_alpha))

        # Sub text
        if t > 0.4:
            sub_alpha = min(255, int(255 * (t - 0.4) / 0.2))
            stw, sth = text_size(od, sub_text, font_sub)
            od.text(((W - stw) // 2, 650), sub_text, font=font_sub, fill=(180, 180, 180, sub_alpha))

        # "9 Parts • In-Depth • Production-Ready"
        if t > 0.5:
            tag_alpha = min(255, int(255 * (t - 0.5) / 0.2))
            tag_font = get_font(24, bold=False)
            tags = "9 Parts  •  In-Depth  •  Production-Ready"
            tagw, tagh = text_size(od, tags, tag_font)
            od.text(((W - tagw) // 2, 720), tags, font=tag_font, fill=(150, 150, 150, tag_alpha))

        img = Image.alpha_composite(img, overlay)
        img = draw_decorative_lines(img)
        frames.append(img.convert("RGB"))

    return frames


# ===================== EASING FUNCTIONS =====================

def ease_out_cubic(t):
    return 1 - (1 - t) ** 3

def ease_in_cubic(t):
    return t ** 3

def ease_out_back(t):
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


# ===================== MAIN =====================

def main():
    print("Generating base background...")
    base_img = make_base_fast()

    print("Generating Frame 1: Logo reveal (2s)...")
    all_frames = generate_frame1_frames(base_img)

    print("Generating Frame 2: Title typing (2s)...")
    all_frames += generate_frame2_frames(base_img)

    print("Generating Frame 3: Parts scroll (9s)...")
    all_frames += generate_frame3_frames(base_img)

    print("Generating Frame 4: CTA (3s)...")
    all_frames += generate_frame4_frames(base_img)

    total_duration = len(all_frames) / FPS
    print(f"Total frames: {len(all_frames)}, Duration: {total_duration:.1f}s")

    # Optimize: reduce to ~10 FPS for GIF by taking every other frame
    print("Optimizing - reducing frame rate for GIF...")
    skip = 2
    optimized_frames = all_frames[::skip]
    frame_duration = int(1000 / FPS * skip)  # ms per frame

    print(f"Optimized frames: {len(optimized_frames)}, Frame duration: {frame_duration}ms")
    print(f"Saving GIF to {OUTPUT_PATH}...")

    # Further optimize by quantizing colors
    optimized_frames[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=optimized_frames[1:],
        duration=frame_duration,
        loop=0,
        optimize=True,
    )

    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"Done! File size: {file_size / 1024 / 1024:.1f} MB")

    if file_size > 10 * 1024 * 1024:
        print("File is large. Creating a more compressed version...")
        # Reduce resolution and colors for smaller file
        smaller_frames = []
        for f in optimized_frames:
            small = f.resize((540, 540), Image.LANCZOS)
            small = small.quantize(colors=128, method=2).convert("RGB")
            smaller_frames.append(small)

        small_path = OUTPUT_PATH.replace(".gif", "_small.gif")
        smaller_frames[0].save(
            small_path,
            save_all=True,
            append_images=smaller_frames[1:],
            duration=frame_duration,
            loop=0,
            optimize=True,
        )
        small_size = os.path.getsize(small_path)
        print(f"Smaller version: {small_size / 1024 / 1024:.1f} MB at {small_path}")


if __name__ == "__main__":
    main()
