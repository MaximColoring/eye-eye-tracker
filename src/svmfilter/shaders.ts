export const gradientResponseVS = [
    "attribute vec2 a_texCoord;",
    "attribute vec2 a_position;",
    "",
    "varying vec2 v_texCoord;",
    "",
    "void main() {",
    "   // transform coordinates to regular coordinates",
    "   gl_Position = vec4(a_position,0.0,1.0);",
    " ",
    "   // pass the texCoord to the fragment shader",
    "   v_texCoord = a_texCoord;",
    "}"
].join("\n")

export const gradientResponseFS = (opp: number[]) =>
    [
        "precision mediump float;",
        "",
        "uniform vec2 u_onePixelPatches;",
        "",
        "// our patches",
        "uniform sampler2D u_patches;",
        "",
        "// the texCoords passed in from the vertex shader.",
        "varying vec2 v_texCoord;",
        "",
        "void main() {",
        "  vec4 bottomLeft = texture2D(u_patches, v_texCoord + vec2(-" +
            opp[0].toFixed(5) +
            ", " +
            opp[1].toFixed(5) +
            "));",
        "  vec4 bottomRight = texture2D(u_patches, v_texCoord + vec2(" +
            opp[0].toFixed(5) +
            ", " +
            opp[1].toFixed(5) +
            "));",
        "  vec4 topLeft = texture2D(u_patches, v_texCoord + vec2(-" +
            opp[0].toFixed(5) +
            ", -" +
            opp[1].toFixed(5) +
            "));",
        "  vec4 topRight = texture2D(u_patches, v_texCoord + vec2(" +
            opp[0].toFixed(5) +
            ", -" +
            opp[1].toFixed(5) +
            "));",
        "  vec4 dx = (",
        "    bottomLeft +",
        "    (texture2D(u_patches, v_texCoord + vec2(-" + opp[0].toFixed(5) + ", 0.0))*vec4(2.0,2.0,2.0,2.0)) +",
        "    topLeft -",
        "    bottomRight -",
        "    (texture2D(u_patches, v_texCoord + vec2(" + opp[0].toFixed(5) + ", 0.0))*vec4(2.0,2.0,2.0,2.0)) -",
        "    topRight)/4.0;",
        "  vec4 dy = (",
        "    bottomLeft +",
        "    (texture2D(u_patches, v_texCoord + vec2(0.0, " + opp[1].toFixed(5) + "))*vec4(2.0,2.0,2.0,2.0)) +",
        "    bottomRight -",
        "    topLeft -",
        "    (texture2D(u_patches, v_texCoord + vec2(0.0, -" + opp[1].toFixed(5) + "))*vec4(2.0,2.0,2.0,2.0)) -",
        "    topRight)/4.0;",
        "  vec4 gradient = sqrt((dx*dx) + (dy*dy));",
        "  gl_FragColor = gradient;",
        "}"
    ].join("\n")

export const lbpResponseVS = [
    "attribute vec2 a_texCoord;",
    "attribute vec2 a_position;",
    "",
    "varying vec2 v_texCoord;",
    "",
    "void main() {",
    "   // transform coordinates to regular coordinates",
    "   gl_Position = vec4(a_position,0.0,1.0);",
    " ",
    "   // pass the texCoord to the fragment shader",
    "   v_texCoord = a_texCoord;",
    "}"
].join("\n")

export const lbpResponseFS = (opp: number[]) =>
    [
        "precision mediump float;",
        "",
        "uniform vec2 u_onePixelPatches;",
        "",
        "// our patches",
        "uniform sampler2D u_patches;",
        "",
        "// the texCoords passed in from the vertex shader.",
        "varying vec2 v_texCoord;",
        "",
        "void main() {",
        "  vec4 topLeft = texture2D(u_patches, v_texCoord + vec2(-" +
            opp[0].toFixed(5) +
            ", -" +
            opp[1].toFixed(5) +
            "));",
        "  vec4 topMid = texture2D(u_patches, v_texCoord + vec2(0.0, -" + opp[1].toFixed(5) + "));",
        "  vec4 topRight = texture2D(u_patches, v_texCoord + vec2(" +
            opp[0].toFixed(5) +
            ", -" +
            opp[1].toFixed(5) +
            "));",
        "  vec4 midLeft = texture2D(u_patches, v_texCoord + vec2(-" + opp[0].toFixed(5) + ", 0.0));",
        "  vec4 midMid = texture2D(u_patches, v_texCoord);",
        "  vec4 midRight = texture2D(u_patches, v_texCoord + vec2(" + opp[0].toFixed(5) + ", 0.0));",
        "  vec4 bottomLeft = texture2D(u_patches, v_texCoord + vec2(-" +
            opp[0].toFixed(5) +
            ", " +
            opp[1].toFixed(5) +
            "));",
        "  vec4 bottomMid = texture2D(u_patches, v_texCoord + vec2(0.0, " + opp[1].toFixed(5) + "));",
        "  vec4 bottomRight = texture2D(u_patches, v_texCoord + vec2(" +
            opp[0].toFixed(5) +
            ", " +
            opp[1].toFixed(5) +
            "));",
        "  vec4 lbp = step(midMid, midRight)*1.0 + step(midMid, topRight)*2.0 + step(midMid, topMid)*4.0;",
        "  lbp = lbp + step(midMid, topLeft)*8.0 + step(midMid, midLeft)*16.0 + step(midMid, bottomLeft)*32.0;",
        "  lbp = lbp + step(midMid, bottomMid)*64.0 + step(midMid, bottomRight)*128.0;",
        "  gl_FragColor = lbp;",
        "}"
    ].join("\n")

export const patchResponseVS = (width: number, height: number, numBlocks: number) =>
    [
        "attribute vec2 a_texCoord;",
        "attribute vec2 a_position;",
        "",
        "const vec2 u_resolution = vec2(" + width.toFixed(1) + "," + height.toFixed(1) + ");",
        "const float u_patchHeight = " + (1 / numBlocks).toFixed(10) + ";",
        "const float u_filterHeight = " + (1 / numBlocks).toFixed(10) + ";",
        "const vec2 u_midpoint = vec2(0.5 ," + (1 / (numBlocks * 2)).toFixed(10) + ");",
        "",
        "varying vec2 v_texCoord;",
        "varying vec2 v_texCoordFilters;",
        "",
        "void main() {",
        "   // convert the rectangle from pixels to 0.0 to 1.0",
        "   vec2 zeroToOne = a_position / u_resolution;",
        "",
        "   // convert from 0->1 to 0->2",
        "   vec2 zeroToTwo = zeroToOne * 2.0;",
        "",
        "   // convert from 0->2 to -1->+1 (clipspace)",
        "   vec2 clipSpace = zeroToTwo - 1.0;",
        "   ",
        "   // transform coordinates to regular coordinates",
        "   gl_Position = vec4(clipSpace * vec2(1.0, 1.0), 0, 1);",
        " ",
        "   // pass the texCoord to the fragment shader",
        "   v_texCoord = a_texCoord;",
        "   ",
        "   // set the filtertexture coordinate based on number filter to use",
        "   v_texCoordFilters = u_midpoint + vec2(0.0, u_filterHeight * floor(a_texCoord[1]/u_patchHeight));",
        "}"
    ].join("\n")

export const patchResponseFS = (
    patchWidth: number,
    patchHeight: number,
    filterWidth: number,
    filterHeight: number,
    numBlocks: number
) =>
    [
        "precision mediump float;",
        "",
        "const vec2 u_onePixelPatches = vec2(" +
            (1 / patchWidth).toFixed(10) +
            "," +
            (1 / (patchHeight * numBlocks)).toFixed(10) +
            ");",
        "const vec2 u_onePixelFilters = vec2(" +
            (1 / filterWidth).toFixed(10) +
            "," +
            (1 / (filterHeight * numBlocks)).toFixed(10) +
            ");",
        "const float u_halffilterwidth = " + ((filterWidth - 1.0) / 2).toFixed(1) + ";",
        "const float u_halffilterheight = " + ((filterHeight - 1.0) / 2).toFixed(1) + ";",
        "",
        "// our patches",
        "uniform sampler2D u_patches;",
        "// our filters",
        "uniform sampler2D u_filters;",
        "",
        "// the texCoords passed in from the vertex shader.",
        "varying vec2 v_texCoord;",
        "varying vec2 v_texCoordFilters; // this should give us correct filter",
        "",
        "void main() {",
        "  vec4 colorSum = vec4(0.0, 0.0, 0.0, 0.0);",
        "  vec4 maxn = vec4(0.0, 0.0, 0.0, 0.0);",
        "  vec4 minn = vec4(256.0, 256.0, 256.0, 256.0);",
        "  vec4 scale = vec4(0.0, 0.0, 0.0, 0.0);",
        "  vec4 patchValue = vec4(0.0, 0.0, 0.0, 0.0);",
        "  vec4 filterValue = vec4(0.0, 0.0, 0.0, 0.0);",
        "  vec4 filterTemp = vec4(0.0, 0.0, 0.0, 0.0);",
        "  for (int w = 0;w < " + filterWidth + ";w++) {",
        "    for (int h = 0;h < " + filterHeight + ";h++) {",
        // tslint:disable-next-line
        "      patchValue = texture2D(u_patches, v_texCoord + u_onePixelPatches * vec2(float(w)-u_halffilterwidth, float(h)-u_halffilterheight));",
        // tslint:disable-next-line
        "      filterValue = texture2D(u_filters, v_texCoordFilters + u_onePixelFilters * vec2(float(w)-u_halffilterwidth, float(h)-u_halffilterheight));",
        "      maxn = max(patchValue, maxn);",
        "      minn = min(patchValue, minn);",
        "      colorSum += patchValue*filterValue;",
        "      filterTemp += filterValue;",
        "    } ",
        "  }",
        "  scale = maxn-minn;",
        "  colorSum = (colorSum-(minn*filterTemp))/scale;",
        "  // logistic transformation",
        "  colorSum = 1.0/(1.0 + exp(- (colorSum) ));",
        "  gl_FragColor = colorSum;",
        "}"
    ].join("\n")

export const drawResponsesVS = [
    "attribute vec2 a_texCoord_draw;",
    "attribute vec2 a_position_draw;",
    "attribute float a_patchChoice_draw;",
    "",
    "uniform vec2 u_resolutiondraw;",
    "",
    "varying vec2 v_texCoord;",
    "varying float v_select;",
    "",
    "void main() {",
    "   // convert the rectangle from pixels to 0.0 to 1.0",
    "   vec2 zeroToOne = a_position_draw / u_resolutiondraw;",
    "",
    "   // convert from 0->1 to 0->2",
    "   vec2 zeroToTwo = zeroToOne * 2.0;",
    "",
    "   // convert from 0->2 to -1->+1 (clipspace)",
    "   vec2 clipSpace = zeroToTwo - 1.0;",
    "   ",
    "   // transform coordinates to regular coordinates",
    "   gl_Position = vec4(clipSpace * vec2(1.0, 1.0), 0, 1);",
    "",
    "   // pass the texCoord to the fragment shader",
    "   v_texCoord = a_texCoord_draw;",
    "   ",
    "   v_select = a_patchChoice_draw;",
    "}"
].join("\n")

export const drawResponsesFS = [
    "precision mediump float;",
    "",
    "// our responses",
    "uniform sampler2D u_responses;",
    "",
    "// the texCoords passed in from the vertex shader.",
    "varying vec2 v_texCoord;",
    "varying float v_select;",
    "",
    "const vec4 bit_shift = vec4(256.0*256.0*256.0, 256.0*256.0, 256.0, 1.0);",
    "const vec4 bit_mask  = vec4(0.0, 1.0/256.0, 1.0/256.0, 1.0/256.0);",
    "",
    "// packing code from here",
    "// http://stackoverflow.com/questions/9882716/packing-float-into-vec4-how-does-this-code-work",
    "void main() {",
    "  vec4 colorSum = texture2D(u_responses, v_texCoord);",
    "  float value = 0.0;",
    "  if (v_select < 0.1) {",
    "    value = colorSum[0];",
    "  } else if (v_select > 0.9 && v_select < 1.1) {",
    "    value = colorSum[1];",
    "  } else if (v_select > 1.9 && v_select < 2.1) {",
    "    value = colorSum[2];",
    "  } else if (v_select > 2.9 && v_select < 3.1) {",
    "    value = colorSum[3];",
    "  } else {",
    "    value = 1.0;",
    "  }",
    "  ",
    "  vec4 res = fract(value * bit_shift);",
    "  res -= res.xxyz * bit_mask;",
    "  ",
    "  //gl_FragColor = vec4(value, value, value, value);",
    "  //gl_FragColor = vec4(1.0, value, 1.0, 1.0);",
    "  gl_FragColor = res;",
    "}"
].join("\n")
