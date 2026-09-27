"""The interface's design rules, enforced on the real stylesheets and scripts: dark and near-black by default, restrained, no blue and no purple,
readable, colours only in the tokens file, one radius, no shadows or gradients, and accessible names on every control."""

import colorsys
import re

import pytest

import web_sources as W

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


def block(start: str) -> dict:
    """The colour custom properties declared in the first block that follows `start` in tokens.css."""
    i = W.TOKENS.index(start)
    body = W.TOKENS[W.TOKENS.index("{", i) + 1: W.TOKENS.index("}", i)]
    return dict(re.findall(r"--([a-z0-9-]+):\s*(#[0-9a-fA-F]{6})", body))


DARK = block(':root[data-theme="dark"]')
LIGHT = block(':root[data-theme="light"]')
THEMES = [("dark", DARK), ("light", LIGHT)]
TEXT = ("ink", "muted", "accent", "danger", "warn")
TOKS = ("tok-keyword", "tok-string", "tok-number", "tok-comment", "tok-function", "tok-punct")


def test_both_themes_define_the_same_palette():
    assert set(DARK) == set(LIGHT) and {"bg", "bg-code", "ink", "muted", "rule", "accent", "mark", "mark-2", "danger", "warn", *TOKS} <= set(DARK)


def test_dark_is_the_default_and_the_choice_is_applied_before_first_paint():
    assert 'data-theme="dark"' in W.HTML.split("<body")[0]
    assert re.search(r":root,\s*:root\[data-theme=\"dark\"\]\s*\{", W.TOKENS)                 # dark values are the unconditional defaults
    theme_js = (W.WEB / "js" / "theme.js").read_text()
    assert "localStorage" in theme_js and "data-theme" in theme_js and "try" in theme_js       # remembered, and safe when storage is blocked
    assert '<script src="/js/theme.js"></script>' in W.HTML.split("</head>")[0]                # a classic, blocking script in <head>, so no flash


def test_no_colour_anywhere_is_blue_or_purple():
    """Hue 170-345 degrees covers cyan, blue, indigo, violet, purple and magenta. Neutrals are near-grey; the accent is green."""
    svg = (W.WEB / "favicon.svg").read_text()
    bad = []
    for h in set(HEX.findall(W.TOKENS + svg)):
        hue, light, sat = (colorsys.rgb_to_hls(*rgb(h))[k] for k in (0, 1, 2))
        if sat > 0.10 and 0.04 < light < 0.96 and 170 <= hue * 360 <= 345:
            bad.append((h, round(hue * 360)))
    assert bad == [], bad
    for name, pal in THEMES:
        assert 100 <= colorsys.rgb_to_hls(*rgb(pal["accent"]))[0] * 360 <= 160, name          # unmistakably green, not teal


def test_neutrals_are_warm_never_bluish():
    for name in ("bg", "bg-code", "bg-hover", "ink", "muted", "rule", "rule-strong"):
        for theme, pal in THEMES:
            r, g, b = rgb(pal[name])
            assert r >= g >= b, (theme, name, pal[name])                                       # blue is never the largest or the middle channel


def test_the_page_background_is_near_black_in_dark_and_off_white_in_light():
    assert luminance(DARK["bg"]) < 0.01 and luminance(LIGHT["bg"]) > 0.9
    assert luminance(DARK["ink"]) > 0.6                                                        # off-white text


@pytest.mark.parametrize("theme,pal", THEMES)
def test_text_colours_meet_wcag_aa_on_their_surfaces(theme, pal):
    for name in TEXT:
        assert contrast(pal[name], pal["bg"]) >= 4.5, (theme, name, contrast(pal[name], pal["bg"]))
    for name in TOKS + ("ink", "muted"):
        assert contrast(pal[name], pal["bg-code"]) >= 4.5, (theme, name, contrast(pal[name], pal["bg-code"]))    # code and comments are readable
    assert contrast(pal["accent-ink"], pal["accent"]) >= 4.5, theme                             # the label on a primary button
    for name in ("mark", "mark-2"):
        assert contrast(pal[name], pal["bg"]) >= 3.0, (theme, name)                             # chart marks need 3:1
    assert contrast(pal["rule"], pal["bg"]) >= 1.1 and contrast(pal["rule-strong"], pal["bg"]) >= 1.4   # hairlines visible but recessive


def test_chart_colours_are_the_ones_that_passed_the_palette_validator():
    """scripts/validate_palette.js (dataviz skill): dark #4fa878 + #c4803f on #121211, light #2f8558 + #a8611f on #fcfcfb pass every hard check;
    the colour-vision separation of the pair is a warning that the charts answer with marker shapes and direct labels."""
    assert (DARK["mark"].lower(), DARK["mark-2"].lower(), DARK["bg"].lower()) == ("#4fa878", "#c4803f", "#121211")
    assert (LIGHT["mark"].lower(), LIGHT["mark-2"].lower(), LIGHT["bg"].lower()) == ("#2f8558", "#a8611f", "#fcfcfb")
    charts = (W.WEB / "js" / "charts.js").read_text()
    assert 'shape: "diamond"' in (W.WEB / "js" / "pages" / "evaluations.js").read_text() and "legend" in charts   # shape and legend, not colour alone


def test_colours_come_from_the_tokens_file_only():
    other = "\n".join(p.read_text() for p in W.CSS_FILES if p.name != "tokens.css")
    assert not HEX.findall(other) and not re.search(r"\b(rgb|hsl)a?\(", other), "a colour is defined outside css/tokens.css"
    assert not HEX.findall(W.JS) and "style.color" not in W.JS and "style.background" not in W.JS and "setAttribute(\"style\"" not in W.JS


def test_restraint_no_shadows_gradients_blur_or_stray_radii():
    for banned in ("box-shadow", "text-shadow", "gradient(", "backdrop-filter", "blur("):
        assert banned not in W.CSS, banned
    radii = set(re.findall(r"border-radius:\s*([^;]+);", W.CSS))
    assert radii <= {"var(--radius)", "0", "50%", "999px", "4px", "3px", "2px"}, radii            # one radius token; pills, dots and tiny details only
    assert len(re.findall(r"--radius:", W.TOKENS)) == 1
    fills = set(re.findall(r"(?<![-\w])background(?:-color)?:\s*([^;]+);", W.CSS))
    assert all(f.startswith("var(--") or f in ("none", "transparent") for f in fills), fills   # every fill is a token
    assert not re.search(r"\.(card|panel|tile|box)\b", W.CSS, re.I)                             # sections are unboxed: type, space, hairlines
    assert not re.search(r"class:\s*[\"'][^\"']*\b(card|panel|tile|box)\b", W.JS, re.I)


def test_script_only_sets_layout_properties_on_style():
    assert set(re.findall(r"\.style\.(\w+)", W.JS)) <= {"left", "top", "height", "width"}


def test_motion_is_subtle_and_respects_the_reduced_motion_setting():
    assert "prefers-reduced-motion" in W.CSS and "animation-duration: 0.001ms" in W.CSS
    durations = [int(x) for x in re.findall(r"--dur:\s*(\d+)ms", W.TOKENS)]
    assert durations and max(durations) <= 200


def test_the_interface_is_keyboard_and_screen_reader_friendly():
    assert ":focus-visible" in W.CSS
    assert 'class="skip"' in W.HTML and 'lang="en"' in W.HTML and 'id="announcer"' in W.HTML and "aria-live" in W.HTML
    assert 'role="region"' in W.HTML and '<main id="view" tabindex="-1">' in W.HTML                # focus moves to the page after navigation
    for feature in ("role: \"tablist\"", "ArrowRight", "aria-expanded", "aria-pressed", "role: \"menu\"", "showModal", "aria-current",
                    "aria-labelledby", "scope: \"col\"", "caption", "role: \"alert\"", "aria-busy"):
        assert feature in W.JS, feature


def _object_literals(source: str, call: str):
    """Every `call({ ... })` object literal in `source`, found by brace matching."""
    for m in re.finditer(r"(?<!function )" + re.escape(call) + r"\(\{", source):
        depth, i = 1, m.end()
        while i < len(source) and depth:
            depth += {"{": 1, "}": -1}.get(source[i], 0)
            i += 1
        yield source[m.end(): i - 1]


def test_every_icon_only_button_has_an_accessible_name():
    """A Button with no visible label draws only an icon, so it must say what it does."""
    checked = 0
    for path in W.JS_FILES:
        for body in _object_literals(path.read_text(), "Button"):
            top = re.sub(r"\{[^{}]*\}", "", body)                                               # ignore nested handlers
            if not re.search(r"\blabel:", top):
                checked += 1
                assert "ariaLabel:" in body, (path.name, body[:80])
    assert checked >= 3


def test_forms_have_labels_and_inputs_are_built_by_the_component_library():
    ui = (W.WEB / "js" / "ui.js").read_text()
    assert 'el("label", { class: "label", for: id }' in ui                                       # Input, Select, Slider, Textarea share one labelled shell
    for path in W.JS_FILES:
        if path.name not in ("ui.js", "playground.js", "landing.js", "charts.js"):
            assert not re.search(r'el\("(input|select|textarea)"', path.read_text()), path.name  # pages use Input/Select/Slider
