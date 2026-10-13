"""Convert CFF-flavoured OpenType fonts to TrueType so matplotlib embeds them
as Type 42 without warnings (ACL's checker rejects Type 3 fonts).
Adapted from the fontTools 'otf2ttf' snippet."""
import sys
from pathlib import Path

from fontTools.ttLib import TTFont, newTable
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen

MAX_ERR = 1.0
POST_FORMAT = 2.0
REVERSE_DIRECTION = True


def glyphs_to_quadratic(glyphs, max_err=MAX_ERR, reverse_direction=REVERSE_DIRECTION):
    quad = {}
    for name in glyphs.keys():
        glyph = glyphs[name]
        tt_pen = TTGlyphPen(glyphs)
        cu2qu_pen = Cu2QuPen(tt_pen, max_err, reverse_direction=reverse_direction)
        glyph.draw(cu2qu_pen)
        quad[name] = tt_pen.glyph()
    return quad


def otf_to_ttf(src, dst):
    ttFont = TTFont(src)
    assert ttFont.sfntVersion == "OTTO"
    glyph_order = ttFont.getGlyphOrder()
    ttFont["loca"] = newTable("loca")
    ttFont["glyf"] = glyf = newTable("glyf")
    glyf.glyphOrder = glyph_order
    glyf.glyphs = glyphs_to_quadratic(ttFont.getGlyphSet())
    del ttFont["CFF "]
    if "VORG" in ttFont:
        del ttFont["VORG"]
    glyf.compile(ttFont)
    ttFont["maxp"] = maxp = newTable("maxp")
    maxp.tableVersion = 0x00010000
    maxp.maxZones = 1
    maxp.maxTwilightPoints = 0
    maxp.maxStorage = 0
    maxp.maxFunctionDefs = 0
    maxp.maxInstructionDefs = 0
    maxp.maxStackElements = 0
    maxp.maxSizeOfInstructions = 0
    maxp.maxComponentElements = max(
        len(g.components if hasattr(g, "components") else []) for g in glyf.glyphs.values())
    maxp.compile(ttFont)
    post = ttFont["post"]
    post.formatType = POST_FORMAT
    post.extraNames = []
    post.mapping = {}
    post.glyphOrder = glyph_order
    ttFont.sfntVersion = "\000\001\000\000"
    ttFont.save(dst)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for src in sys.argv[2:]:
        dst = out / (Path(src).stem + ".ttf")
        otf_to_ttf(src, str(dst))
        print("wrote", dst)
