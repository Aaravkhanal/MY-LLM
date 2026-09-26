"""The interface's design rules, enforced on the real stylesheet: minimal, no boxes, no blue, no purple, readable."""

import colorsys
import re
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "src" / "aarav_lm" / "web"
CSS = (WEB / "style.css").read_text()
JS = (WEB / "app.js").read_text()
HEX = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b")


def rgb(h: str):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def luminance(h: str) -> float:
    lin = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4          # noqa: E731
    r, g, b = (lin(c) for c in rgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def theme(block_start: str) -> dict:
    """The custom properties declared in the first block that follows `block_start`."""
    i = CSS.index(block_start)
    body = CSS[CSS.index("{", i) + 1: CSS.index("}", i)]
    return dict(re.findall(r"--([a-z-]+):\s*(#[0-9a-fA-F]{6})", body))


LIGHT = theme(":root {")
DARK = theme(':root[data-theme="dark"]')
MEDIA_DARK = theme(':root:not([data-theme="light"])')


def test_both_themes_and_the_automatic_dark_mode_define_the_same_palette():
    keys = {"bg", "ink", "muted", "rule", "accent", "mark", "danger"}
    assert keys <= set(LIGHT) and keys <= set(DARK) and keys <= set(MEDIA_DARK)
    assert DARK == MEDIA_DARK                                        # the OS setting and the explicit toggle look identical


def test_no_colour_in_the_stylesheet_is_blue_or_purple():
    """Hue 170-345 degrees covers cyan, blue, indigo, violet, purple and magenta. Warm neutrals are near-grey; the accent is green."""
    bad = []
    for h in set(HEX.findall(CSS)):
        hue, light, sat = colorsys.rgb_to_hls(*rgb(h))[0] * 360, colorsys.rgb_to_hls(*rgb(h))[1], colorsys.rgb_to_hls(*rgb(h))[2]
        chromatic = sat > 0.10 and 0.04 < light < 0.96
        if chromatic and 170 <= hue <= 345:
            bad.append((h, round(hue)))
    assert bad == [], bad
    accent_hue = colorsys.rgb_to_hls(*rgb(LIGHT["accent"]))[0] * 360
    assert 100 <= accent_hue <= 160                                  # unmistakably green, not teal


def test_neutrals_are_warm_never_bluish():
    for name in ("bg", "ink", "muted", "rule"):
        for palette in (LIGHT, DARK):
            r, g, b = rgb(palette[name])
            assert r >= g >= b, (name, palette[name])                # blue is never the largest or the middle channel


@pytest.mark.parametrize("palette_name,palette", [("light", LIGHT), ("dark", DARK)])
def test_text_colours_meet_wcag_aa_on_the_page_background(palette_name, palette):
    for name in ("ink", "muted", "accent", "danger"):
        assert contrast(palette[name], palette["bg"]) >= 4.5, (palette_name, name, contrast(palette[name], palette["bg"]))
    assert contrast(palette["mark"], palette["bg"]) >= 3.0            # chart marks need 3:1
    assert contrast(palette["rule"], palette["bg"]) >= 1.1            # hairlines visible but recessive


def test_chart_colours_are_the_ones_that_passed_the_palette_validator():
    """scripts/validate_palette.js (dataviz skill): #2f8558 on #fcfcfb (light) and #4fa878 on #1a1a19 (dark) pass all checks."""
    assert LIGHT["mark"].lower() == "#2f8558" and DARK["mark"].lower() == "#4fa878"
    assert LIGHT["bg"].lower() == "#fcfcfb" and DARK["bg"].lower() == "#1a1a19"


def test_there_are_no_boxes_shadows_or_filled_panels():
    assert "box-shadow" not in CSS and "text-shadow" not in CSS
    assert set(re.findall(r"border-radius:\s*([^;]+);", CSS)) <= {"0"}   # square and flat: rounding lives only on chart bar ends (SVG)
    fills = set(re.findall(r"(?<![-\w])background(?:-color)?:\s*([^;]+);", CSS))
    assert fills <= {"var(--bg)", "none", "transparent"}, fills       # nothing is a tinted panel
    for border in re.findall(r"(?<![-\w])border:\s*([^;]+);", CSS):
        assert border.strip() == "0", border                          # no all-sides borders: only hairlines on one edge
    assert not re.search(r"\bcard\b|\bpanel\b|\bmodal\b", CSS + JS, re.I)


def test_colours_come_from_the_stylesheet_only():
    assert not HEX.findall(JS)
    assert "style.color" not in JS and "style.background" not in JS


def test_the_interface_follows_the_system_theme_and_respects_reduced_motion():
    assert "prefers-color-scheme: dark" in CSS and "prefers-reduced-motion" in CSS
    assert 'name="color-scheme"' in (WEB / "index.html").read_text()


def test_focus_is_visible_for_keyboard_users_and_forms_are_labelled():
    assert ":focus-visible" in CSS
    html = (WEB / "index.html").read_text()
    assert 'class="skip"' in html and 'lang="en"' in html
    assert JS.count("aria-label") >= 3 and "aria-live" in JS and 'role: "alert"' in JS
