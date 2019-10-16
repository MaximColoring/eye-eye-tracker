import { setupWebGL, loadShader, createProgram } from "./webglutils"
import {
    gradientResponseVS,
    gradientResponseFS,
    lbpResponseVS,
    lbpResponseFS,
    drawResponsesVS,
    drawResponsesFS,
    patchResponseVS,
    patchResponseFS
} from "./shaders"

export const webglFilter = () => {
    /*
     * Textures:
     * 0 : raw filter
     * 1 : patches
     * 2 : finished response
     * 3 : grad/lbp treated patches
     * 4 : sobel filter
     * 5 : lbp filter
     *
     * Routing:
     *         (              )  0/4/5 --\
     *         (              )          _\|
     * 1 ----> ( ---------->3 ) ----------> 2
     *         lbpResponse/      patchResponse
     *         gradientResponse
     */
    let lbpInit = false
    let sobelInit = false

    let biases: FiltersTypes<number[]>
    let filterWidth: number
    let filterHeight: number
    let patchWidth: number
    let patchHeight: number
    let numPatches: number
    let numBlocks: number
    let canvasWidth: number
    let canvasHeight: number
    let newCanvasWidth: number
    let newCanvasBlockHeight: number
    let newCanvasHeight: number
    let patchCells: number
    let textureWidth: number
    let textureHeight: number
    let patchSize: number
    let patchArray: Float32Array
    let opp: number[]

    let canvas: HTMLCanvasElement

    let patchTex: WebGLTexture | null
    let patchDrawProgram: WebGLProgram | null

    let drawRectBuffer: WebGLBuffer | null
    let drawImageBuffer: WebGLBuffer | null
    let drawLayerBuffer: WebGLBuffer | null
    let drawOutRectangles: Float32Array
    let drawOutImages: Float32Array
    let drawOutLayer: Float32Array

    let gl: WebGLRenderingContext
    let patchResponseProgram: WebGLProgram | null
    let apositionBuffer: WebGLBuffer | null
    let texCoordBuffer: WebGLBuffer | null
    let fbo: WebGLFramebuffer | null

    let gradientResponseProgram: WebGLProgram | null
    let gradAPositionBuffer: WebGLBuffer | null
    let gradTexCoordBuffer: WebGLBuffer | null
    let gbo: WebGLFramebuffer | null

    let lbpResponseProgram: WebGLProgram | null
    let lbpAPositionBuffer: WebGLBuffer | null
    let lbpTexCoordBuffer: WebGLBuffer | null

    const init = (
        filters: FiltersTypes,
        bias: FiltersTypes<number[]>,
        nP: number,
        pW: number,
        pH: number,
        fW: number,
        fH: number
    ) => {
        // we assume filterVector goes from left to right, rowwise, i.e. row-major order

        if (fW !== fH) {
            alert("filter width and height must be same size!")
            return
        }

        // if filter width is not odd, alert
        if (fW % 2 === 0 || fH % 2 === 0) {
            alert("filters used in svm must be of odd dimensions!")
            return
        }

        // setup variables
        biases = bias
        filterWidth = fW
        filterHeight = fH
        patchWidth = pW
        patchHeight = pH
        numPatches = nP
        numBlocks = Math.floor(numPatches / 4) + Math.ceil((numPatches % 4) / 4)
        canvasWidth = patchWidth
        canvasHeight = patchHeight * numBlocks
        newCanvasWidth = patchWidth - filterWidth + 1
        newCanvasBlockHeight = patchHeight - filterWidth + 1
        newCanvasHeight = newCanvasBlockHeight * numPatches
        patchCells = Math.floor(numPatches / 4) + Math.ceil((numPatches % 4) / 4)
        textureWidth = patchWidth
        textureHeight = patchHeight * patchCells
        patchSize = patchWidth * patchHeight
        patchArray = new Float32Array(patchSize * patchCells * 4)
        opp = [1 / patchWidth, 1 / (patchHeight * numBlocks)]

        // create webglcanvas
        canvas = document.createElement("canvas")
        canvas.setAttribute("width", patchWidth - filterWidth + 1 + "px")
        canvas.setAttribute("height", (patchHeight - filterHeight + 1) * numPatches + "px")
        canvas.setAttribute("id", "renderCanvas")
        canvas.setAttribute("style", "display:none;")
        // document.body.appendChild(canvas);
        gl = setupWebGL(canvas, {
            premultipliedAlpha: false,
            preserveDrawingBuffer: true,
            antialias: false
        })!

        if (!gl) return

        // check for float textures support and fail if not
        if (!gl.getExtension("OES_texture_float")) {
            alert("Your graphics card does not support floating point textures! :(")
            return
        }

        // insert filters into textures
        if ("raw" in filters) {
            insertFilter(filters.raw, gl.TEXTURE0)
        }
        if ("sobel" in filters) {
            insertFilter(filters.sobel, gl.TEXTURE4)
            sobelInit = true
        }
        if ("lbp" in filters) {
            insertFilter(filters.lbp, gl.TEXTURE5)
            lbpInit = true
        }

        // calculate vertices for calculating responses

        // vertex rectangles to draw out
        let rs: number[] = []
        const halfFilter = (filterWidth - 1) / 2
        let yOffset: number
        for (let i = 0; i < numBlocks; i++) {
            yOffset = i * patchHeight
            // first triangle
            rs = rs.concat([
                halfFilter,
                yOffset + halfFilter,
                patchWidth - halfFilter,
                yOffset + halfFilter,
                halfFilter,
                yOffset + patchHeight - halfFilter
            ])
            // second triangle
            rs = rs.concat([
                halfFilter,
                yOffset + patchHeight - halfFilter,
                patchWidth - halfFilter,
                yOffset + halfFilter,
                patchWidth - halfFilter,
                yOffset + patchHeight - halfFilter
            ])
        }
        const rectangles = new Float32Array(rs)

        // image rectangles to draw out
        const irs: number[] = []
        for (let i = 0; i < rectangles.length; i++) {
            irs[i] = i % 2 === 0 ? rectangles[i] / canvasWidth : rectangles[i] / canvasHeight
        }
        const irectangles = new Float32Array(irs)

        // vertices for drawing out responses

        // drawOutRectangles
        drawOutRectangles = new Float32Array(12 * numPatches)
        let indexOffset
        for (let i = 0; i < numPatches; i++) {
            yOffset = i * newCanvasBlockHeight
            indexOffset = i * 12

            // first triangle
            drawOutRectangles[indexOffset] = 0.0
            drawOutRectangles[indexOffset + 1] = yOffset
            drawOutRectangles[indexOffset + 2] = newCanvasWidth
            drawOutRectangles[indexOffset + 3] = yOffset
            drawOutRectangles[indexOffset + 4] = 0.0
            drawOutRectangles[indexOffset + 5] = yOffset + newCanvasBlockHeight

            // second triangle
            drawOutRectangles[indexOffset + 6] = 0.0
            drawOutRectangles[indexOffset + 7] = yOffset + newCanvasBlockHeight
            drawOutRectangles[indexOffset + 8] = newCanvasWidth
            drawOutRectangles[indexOffset + 9] = yOffset
            drawOutRectangles[indexOffset + 10] = newCanvasWidth
            drawOutRectangles[indexOffset + 11] = yOffset + newCanvasBlockHeight
        }

        // images
        drawOutImages = new Float32Array(numPatches * 12)
        const halfFilterWidth = (filterWidth - 1) / 2 / patchWidth
        const halfFilterHeight = (filterWidth - 1) / 2 / (patchHeight * patchCells)
        const patchHeightT = patchHeight / (patchHeight * patchCells)
        for (let i = 0; i < numPatches; i++) {
            yOffset = Math.floor(i / 4) * patchHeightT
            indexOffset = i * 12

            // first triangle
            drawOutImages[indexOffset] = halfFilterWidth
            drawOutImages[indexOffset + 1] = yOffset + halfFilterHeight
            drawOutImages[indexOffset + 2] = 1.0 - halfFilterWidth
            drawOutImages[indexOffset + 3] = yOffset + halfFilterHeight
            drawOutImages[indexOffset + 4] = halfFilterWidth
            drawOutImages[indexOffset + 5] = yOffset + patchHeightT - halfFilterHeight

            // second triangle
            drawOutImages[indexOffset + 6] = halfFilterWidth
            drawOutImages[indexOffset + 7] = yOffset + patchHeightT - halfFilterHeight
            drawOutImages[indexOffset + 8] = 1.0 - halfFilterWidth
            drawOutImages[indexOffset + 9] = yOffset + halfFilterHeight
            drawOutImages[indexOffset + 10] = 1.0 - halfFilterWidth
            drawOutImages[indexOffset + 11] = yOffset + patchHeightT - halfFilterHeight
        }

        // layer
        drawOutLayer = new Float32Array(numPatches * 6)
        let layernum
        for (let i = 0; i < numPatches; i++) {
            layernum = i % 4
            indexOffset = i * 6
            drawOutLayer[indexOffset] = layernum
            drawOutLayer[indexOffset + 1] = layernum
            drawOutLayer[indexOffset + 2] = layernum
            drawOutLayer[indexOffset + 3] = layernum
            drawOutLayer[indexOffset + 4] = layernum
            drawOutLayer[indexOffset + 5] = layernum
        }

        // set up programs and load attributes etc

        if ("lbp" in filters || "sobel" in filters) {
            let topCoord = 1.0 - 2 / (patchHeight * numBlocks)
            let bottomCoord = 1.0 - 2 / numBlocks + 2 / (patchHeight * numBlocks)
            // calculate position of vertex rectangles for gradient/lbp program
            let grs: number[] = []
            for (let i = 0; i < numBlocks; i++) {
                yOffset = i * (2 / numBlocks)
                // first triangle
                grs = grs.concat([-1.0, topCoord - yOffset, 1.0, topCoord - yOffset, -1.0, bottomCoord - yOffset])
                // second triangle
                grs = grs.concat([-1.0, bottomCoord - yOffset, 1.0, topCoord - yOffset, 1.0, bottomCoord - yOffset])
            }
            const gradRectangles = new Float32Array(grs)

            topCoord = 1.0 - 1 / (patchHeight * numBlocks)
            bottomCoord = 1.0 - 1 / numBlocks + 1 / (patchHeight * numBlocks)
            // calculate position of image rectangles to draw out
            let girs: number[] = []
            for (let i = 0; i < numBlocks; i++) {
                yOffset = i * (1 / numBlocks)
                // first triangle
                girs = girs.concat([0.0, topCoord - yOffset, 1.0, topCoord - yOffset, 0.0, bottomCoord - yOffset])
                // second triangle
                girs = girs.concat([0.0, bottomCoord - yOffset, 1.0, topCoord - yOffset, 1.0, bottomCoord - yOffset])
            }
            const gradIRectangles = new Float32Array(girs)

            if ("sobel" in filters) {
                const grVertexShader = loadShader(gl, gradientResponseVS, gl.VERTEX_SHADER)
                const grFragmentShader = loadShader(gl, gradientResponseFS(opp), gl.FRAGMENT_SHADER)
                gradientResponseProgram = createProgram(gl, [grVertexShader!, grFragmentShader!])
                gl.useProgram(gradientResponseProgram)

                // set up vertices with rectangles
                const gradPositionLocation = gl.getAttribLocation(gradientResponseProgram!, "a_position")
                gradAPositionBuffer = gl.createBuffer()
                gl.bindBuffer(gl.ARRAY_BUFFER, gradAPositionBuffer)
                gl.bufferData(gl.ARRAY_BUFFER, gradRectangles, gl.STATIC_DRAW)
                gl.enableVertexAttribArray(gradPositionLocation)
                gl.vertexAttribPointer(gradPositionLocation, 2, gl.FLOAT, false, 0, 0)

                // set up texture positions
                const gradTexCoordLocation = gl.getAttribLocation(gradientResponseProgram!, "a_texCoord")
                gradTexCoordBuffer = gl.createBuffer()
                gl.bindBuffer(gl.ARRAY_BUFFER, gradTexCoordBuffer)
                gl.bufferData(gl.ARRAY_BUFFER, gradIRectangles, gl.STATIC_DRAW)
                gl.enableVertexAttribArray(gradTexCoordLocation)
                gl.vertexAttribPointer(gradTexCoordLocation, 2, gl.FLOAT, false, 0, 0)

                // set up patches texture in gradientResponseProgram
                gl.uniform1i(gl.getUniformLocation(gradientResponseProgram!, "u_patches"), 1)
            }
            if ("lbp" in filters) {
                const lbpVertexShader = loadShader(gl, lbpResponseVS, gl.VERTEX_SHADER)
                const lbpFragmentShader = loadShader(gl, lbpResponseFS(opp), gl.FRAGMENT_SHADER)
                lbpResponseProgram = createProgram(gl, [lbpVertexShader!, lbpFragmentShader!])
                gl.useProgram(lbpResponseProgram)

                // set up vertices with rectangles
                const lbpPositionLocation = gl.getAttribLocation(lbpResponseProgram!, "a_position")
                lbpAPositionBuffer = gl.createBuffer()
                gl.bindBuffer(gl.ARRAY_BUFFER, lbpAPositionBuffer)
                gl.bufferData(gl.ARRAY_BUFFER, gradRectangles, gl.STATIC_DRAW)
                gl.enableVertexAttribArray(lbpPositionLocation)
                gl.vertexAttribPointer(lbpPositionLocation, 2, gl.FLOAT, false, 0, 0)

                // set up texture positions
                const lbpTexCoordLocation = gl.getAttribLocation(lbpResponseProgram!, "a_texCoord")
                lbpTexCoordBuffer = gl.createBuffer()
                gl.bindBuffer(gl.ARRAY_BUFFER, lbpTexCoordBuffer)
                gl.bufferData(gl.ARRAY_BUFFER, gradIRectangles, gl.STATIC_DRAW)
                gl.enableVertexAttribArray(lbpTexCoordLocation)
                gl.vertexAttribPointer(lbpTexCoordLocation, 2, gl.FLOAT, false, 0, 0)

                // set up patches texture in lbpResponseProgram
                gl.uniform1i(gl.getUniformLocation(lbpResponseProgram!, "u_patches"), 1)
            }
        }

        // setup patchdraw program
        const drVertexShader = loadShader(gl, drawResponsesVS, gl.VERTEX_SHADER)
        const drFragmentShader = loadShader(gl, drawResponsesFS, gl.FRAGMENT_SHADER)
        patchDrawProgram = createProgram(gl, [drVertexShader!, drFragmentShader!])
        gl.useProgram(patchDrawProgram)

        // set the resolution/dimension of the canvas
        const resolutionLocation = gl.getUniformLocation(patchDrawProgram!, "u_resolutiondraw")
        gl.uniform2f(resolutionLocation, newCanvasWidth, newCanvasHeight)

        // set u_responses
        const responsesLocation = gl.getUniformLocation(patchDrawProgram!, "u_responses")
        gl.uniform1i(responsesLocation, 2)

        // setup patchresponse program
        const prVertexShader = loadShader(gl, patchResponseVS(canvasWidth, canvasHeight, numBlocks), gl.VERTEX_SHADER)
        const prFragmentShader = loadShader(
            gl,
            patchResponseFS(patchWidth, patchHeight, filterWidth, filterHeight, numBlocks),
            gl.FRAGMENT_SHADER
        )
        patchResponseProgram = createProgram(gl, [prVertexShader!, prFragmentShader!])
        gl.useProgram(patchResponseProgram)

        // set up vertices with rectangles
        const positionLocation = gl.getAttribLocation(patchResponseProgram!, "a_position")
        apositionBuffer = gl.createBuffer()
        gl.bindBuffer(gl.ARRAY_BUFFER, apositionBuffer)
        gl.bufferData(gl.ARRAY_BUFFER, rectangles, gl.STATIC_DRAW)
        gl.enableVertexAttribArray(positionLocation)
        gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0)

        // set up texture positions
        const texCoordLocation = gl.getAttribLocation(patchResponseProgram!, "a_texCoord")
        texCoordBuffer = gl.createBuffer()
        gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer)
        gl.bufferData(gl.ARRAY_BUFFER, irectangles, gl.STATIC_DRAW)
        gl.enableVertexAttribArray(texCoordLocation)
        gl.vertexAttribPointer(texCoordLocation, 2, gl.FLOAT, false, 0, 0)

        if ("lbp" in filters || "sobel" in filters) {
            // set up gradient/lbp buffer (also used for lbp)
            gl.activeTexture(gl.TEXTURE3)
            const gradients = gl.createTexture()
            gl.bindTexture(gl.TEXTURE_2D, gradients)
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, patchWidth, patchHeight * numBlocks, 0, gl.RGBA, gl.FLOAT, null)
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)

            // set up gradient/lbp framebuffer
            gbo = gl.createFramebuffer()
            gl.bindFramebuffer(gl.FRAMEBUFFER, gbo)
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, gradients, 0)
        }

        // set up buffer to draw to
        gl.activeTexture(gl.TEXTURE2)
        const rttTexture = gl.createTexture()
        gl.bindTexture(gl.TEXTURE_2D, rttTexture)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, patchWidth, patchHeight * numBlocks, 0, gl.RGBA, gl.FLOAT, null)

        // set up response framebuffer
        fbo = gl.createFramebuffer()
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, rttTexture, 0)

        gl.viewport(0, 0, patchWidth, patchHeight * numBlocks)

        /* initialize some textures and buffers used later on */

        patchTex = gl.createTexture()
        drawRectBuffer = gl.createBuffer()
        drawImageBuffer = gl.createBuffer()
        drawLayerBuffer = gl.createBuffer()
    }

    const getRawResponses = (patches: number[][]): number[][] => {
        if (!gl || !patchResponseProgram) return []

        insertPatches(patches)

        // switch to correct program
        gl.useProgram(patchResponseProgram)

        // set u_patches to point to texture 1
        gl.uniform1i(gl.getUniformLocation(patchResponseProgram, "u_patches"), 1)

        // set u_filters to point to correct filter
        gl.uniform1i(gl.getUniformLocation(patchResponseProgram, "u_filters"), 0)

        // set up vertices with rectangles
        const positionLocation = gl.getAttribLocation(patchResponseProgram, "a_position")
        gl.bindBuffer(gl.ARRAY_BUFFER, apositionBuffer)
        gl.enableVertexAttribArray(positionLocation)
        gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0)

        // set up texture positions
        const texCoordLocation = gl.getAttribLocation(patchResponseProgram, "a_texCoord")
        gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer)
        gl.enableVertexAttribArray(texCoordLocation)
        gl.vertexAttribPointer(texCoordLocation, 2, gl.FLOAT, false, 0, 0)

        // set framebuffer to the original one if not already using it
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)

        gl.viewport(0, 0, patchWidth, patchHeight * numBlocks)

        gl.clearColor(0.0, 0.0, 0.0, 1.0)
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

        // draw to framebuffer
        gl.drawArrays(gl.TRIANGLES, 0, patchCells * 6)

        // gl.finish();

        const responses = drawOut("raw")

        return responses
    }

    const getSobelResponses = (patches: number[][]): number[][] => {
        // check that it is initialized
        if (!sobelInit || !gl || !gradientResponseProgram || !patchResponseProgram) return []

        insertPatches(patches)

        /* do sobel filter on patches */

        // switch to correct program
        gl.useProgram(gradientResponseProgram)

        // set up vertices with rectangles
        const gradPositionLocation = gl.getAttribLocation(gradientResponseProgram, "a_position")
        gl.bindBuffer(gl.ARRAY_BUFFER, gradAPositionBuffer)
        gl.enableVertexAttribArray(gradPositionLocation)
        gl.vertexAttribPointer(gradPositionLocation, 2, gl.FLOAT, false, 0, 0)

        // set up texture positions
        const gradTexCoordLocation = gl.getAttribLocation(gradientResponseProgram, "a_texCoord")
        gl.bindBuffer(gl.ARRAY_BUFFER, gradTexCoordBuffer)
        gl.enableVertexAttribArray(gradTexCoordLocation)
        gl.vertexAttribPointer(gradTexCoordLocation, 2, gl.FLOAT, false, 0, 0)

        // set framebuffer to the original one if not already using it
        gl.bindFramebuffer(gl.FRAMEBUFFER, gbo)

        gl.viewport(0, 0, patchWidth, patchHeight * numBlocks)

        gl.clearColor(0.0, 0.0, 0.0, 1.0)
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

        // draw to framebuffer
        gl.drawArrays(gl.TRIANGLES, 0, patchCells * 6)

        /* calculate responses */

        gl.useProgram(patchResponseProgram)

        // set patches and filters to point to correct textures
        gl.uniform1i(gl.getUniformLocation(patchResponseProgram, "u_filters"), 4)
        gl.uniform1i(gl.getUniformLocation(patchResponseProgram, "u_patches"), 3)

        const positionLocation = gl.getAttribLocation(patchResponseProgram, "a_position")
        gl.bindBuffer(gl.ARRAY_BUFFER, apositionBuffer)
        gl.enableVertexAttribArray(positionLocation)
        gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0)

        // set up texture positions
        const texCoordLocation = gl.getAttribLocation(patchResponseProgram, "a_texCoord")
        gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer)
        gl.enableVertexAttribArray(texCoordLocation)
        gl.vertexAttribPointer(texCoordLocation, 2, gl.FLOAT, false, 0, 0)

        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
        gl.viewport(0, 0, patchWidth, patchHeight * numBlocks)

        gl.clearColor(0.0, 0.0, 0.0, 1.0)
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

        // draw to framebuffer
        gl.drawArrays(gl.TRIANGLES, 0, patchCells * 6)

        /* get the responses */

        const responses = drawOut("sobel")

        return responses
    }

    const getLBPResponses = (patches: number[][]): number[][] => {
        // check that it is initialized
        if (!lbpInit || !gl || !lbpResponseProgram || !patchResponseProgram) return []

        insertPatches(patches)

        /* do sobel filter on patches */

        // switch to correct program
        gl.useProgram(lbpResponseProgram)

        // set up vertices with rectangles
        const lbpPositionLocation = gl.getAttribLocation(lbpResponseProgram, "a_position")
        gl.bindBuffer(gl.ARRAY_BUFFER, lbpAPositionBuffer)
        gl.enableVertexAttribArray(lbpPositionLocation)
        gl.vertexAttribPointer(lbpPositionLocation, 2, gl.FLOAT, false, 0, 0)

        // set up texture positions
        const lbpTexCoordLocation = gl.getAttribLocation(lbpResponseProgram, "a_texCoord")
        gl.bindBuffer(gl.ARRAY_BUFFER, lbpTexCoordBuffer)
        gl.enableVertexAttribArray(lbpTexCoordLocation)
        gl.vertexAttribPointer(lbpTexCoordLocation, 2, gl.FLOAT, false, 0, 0)

        // set framebuffer to the original one if not already using it
        gl.bindFramebuffer(gl.FRAMEBUFFER, gbo)

        gl.viewport(0, 0, patchWidth, patchHeight * numBlocks)

        gl.clearColor(0.0, 0.0, 0.0, 1.0)
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

        // draw to framebuffer
        gl.drawArrays(gl.TRIANGLES, 0, patchCells * 6)

        /* calculate responses */

        gl.useProgram(patchResponseProgram)

        gl.uniform1i(gl.getUniformLocation(patchResponseProgram, "u_filters"), 5)
        gl.uniform1i(gl.getUniformLocation(patchResponseProgram, "u_patches"), 3)

        const positionLocation = gl.getAttribLocation(patchResponseProgram, "a_position")
        gl.bindBuffer(gl.ARRAY_BUFFER, apositionBuffer)
        gl.enableVertexAttribArray(positionLocation)
        gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0)

        // set up texture positions
        const texCoordLocation = gl.getAttribLocation(patchResponseProgram, "a_texCoord")
        gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer)
        gl.enableVertexAttribArray(texCoordLocation)
        gl.vertexAttribPointer(texCoordLocation, 2, gl.FLOAT, false, 0, 0)

        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
        gl.viewport(0, 0, patchWidth, patchHeight * numBlocks)

        gl.clearColor(0.0, 0.0, 0.0, 1.0)
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

        // draw to framebuffer
        gl.drawArrays(gl.TRIANGLES, 0, patchCells * 6)

        /* get the responses */

        const responses = drawOut("lbp")

        return responses
    }

    const insertPatches = (patches: number[][]) => {
        if (!gl) return
        // pass patches into texture, each patch in either r, g, b or a
        let patchArrayIndex = 0
        let patchesIndex1 = 0
        let patchesIndex2 = 0
        for (let i = 0; i < patchCells; i++) {
            for (let j = 0; j < patchHeight; j++) {
                for (let k = 0; k < patchWidth; k++) {
                    patchesIndex1 = i * 4
                    patchesIndex2 = j * patchWidth + k
                    patchArrayIndex = (patchSize * i + patchesIndex2) * 4

                    // set r with first patch
                    patchArray[patchArrayIndex] = patchesIndex1 < numPatches ? patches[patchesIndex1][patchesIndex2] : 0

                    // set g with 2nd patch
                    patchArray[patchArrayIndex + 1] =
                        patchesIndex1 + 1 < numPatches ? patches[patchesIndex1 + 1][patchesIndex2] : 0
                    // set b with 3rd patch
                    patchArray[patchArrayIndex + 2] =
                        patchesIndex1 + 2 < numPatches ? patches[patchesIndex1 + 2][patchesIndex2] : 0
                    // set a with 4th patch
                    patchArray[patchArrayIndex + 3] =
                        patchesIndex1 + 3 < numPatches ? patches[patchesIndex1 + 3][patchesIndex2] : 0
                }
            }
        }

        // pass texture into an uniform
        gl.activeTexture(gl.TEXTURE1)
        gl.bindTexture(gl.TEXTURE_2D, patchTex)
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, textureWidth, textureHeight, 0, gl.RGBA, gl.FLOAT, patchArray)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    }

    const insertFilter = (filter: number[][], textureNum: number) => {
        if (!gl) return
        const filterSize = filterWidth * filterHeight
        const filterArray = new Float32Array(filterSize * numBlocks * 4)
        for (let i = 0; i < numBlocks; i++) {
            for (let j = 0; j < filterHeight; j++) {
                for (let k = 0; k < filterWidth; k++) {
                    // set r with first filter
                    filterArray[(filterSize * i + j * filterWidth + k) * 4] =
                        i * 4 < filter.length ? filter[i * 4][j * filterWidth + k] : 0
                    // set g with 2nd filter
                    filterArray[(filterSize * i + j * filterWidth + k) * 4 + 1] =
                        i * 4 + 1 < filter.length ? filter[i * 4 + 1][j * filterWidth + k] : 0
                    // set b with 3rd filter
                    filterArray[(filterSize * i + j * filterWidth + k) * 4 + 2] =
                        i * 4 + 2 < filter.length ? filter[i * 4 + 2][j * filterWidth + k] : 0
                    // set a with 4th filter
                    filterArray[(filterSize * i + j * filterWidth + k) * 4 + 3] =
                        i * 4 + 3 < filter.length ? filter[i * 4 + 3][j * filterWidth + k] : 0
                }
            }
        }

        gl.activeTexture(textureNum)
        const filterTexture = gl.createTexture()
        gl.bindTexture(gl.TEXTURE_2D, filterTexture)
        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA,
            filterWidth,
            filterHeight * numBlocks,
            0,
            gl.RGBA,
            gl.FLOAT,
            filterArray
        )
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    }

    const drawOut = (type: keyof FiltersTypes): number[][] => {
        if (!patchDrawProgram) return []
        // switch programs
        gl.useProgram(patchDrawProgram)

        // bind canvas buffer
        gl.bindFramebuffer(gl.FRAMEBUFFER, null)
        gl.viewport(0, 0, newCanvasWidth, newCanvasHeight)

        gl.clearColor(0.0, 0.0, 0.0, 1.0)
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

        gl.bindBuffer(gl.ARRAY_BUFFER, drawRectBuffer)
        gl.bufferData(gl.ARRAY_BUFFER, drawOutRectangles, gl.STATIC_DRAW)
        const positionLocation = gl.getAttribLocation(patchDrawProgram, "a_position_draw")
        gl.enableVertexAttribArray(positionLocation)
        gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0)

        gl.bindBuffer(gl.ARRAY_BUFFER, drawImageBuffer)
        gl.bufferData(gl.ARRAY_BUFFER, drawOutImages, gl.STATIC_DRAW)
        const textureLocation = gl.getAttribLocation(patchDrawProgram, "a_texCoord_draw")
        gl.enableVertexAttribArray(textureLocation)
        gl.vertexAttribPointer(textureLocation, 2, gl.FLOAT, false, 0, 0)

        gl.bindBuffer(gl.ARRAY_BUFFER, drawLayerBuffer)
        gl.bufferData(gl.ARRAY_BUFFER, drawOutLayer, gl.STATIC_DRAW)
        const layerLocation = gl.getAttribLocation(patchDrawProgram, "a_patchChoice_draw")
        gl.enableVertexAttribArray(layerLocation)
        gl.vertexAttribPointer(layerLocation, 1, gl.FLOAT, false, 0, 0)

        // draw out
        gl.drawArrays(gl.TRIANGLES, 0, numPatches * 6)

        const responses = addBias(splitArray(unpackToFloat(getOutput() as any), numPatches), biases[type])

        // normalize responses to lie within [0,1]
        const rl = responses.length
        for (let i = 0; i < rl; i++) {
            responses[i] = normalizeFilterMatrix(responses[i])
        }

        return responses
    }

    const addBias = (responses: number[][], bias: number[]) => {
        // do a little trick to add bias in the logit function
        let biasMult
        for (let i = 0; i < responses.length; i++) {
            biasMult = Math.exp(bias[i])
            for (let j = 0; j < responses[i].length; j++) {
                responses[i][j] = 1 / (1 + (1 - responses[i][j]) / (responses[i][j] * biasMult))
            }
        }
        return responses
    }

    const splitArray = (array: number[], parts: number) => {
        const sp = []
        const al = array.length
        const splitlength = al / parts
        let ta = []
        for (let i = 0; i < al; i++) {
            if (i % splitlength === 0) {
                if (i !== 0) {
                    sp.push(ta)
                }
                ta = []
            }
            ta.push(array[i])
        }
        sp.push(ta)
        return sp
    }

    const getOutput = () => {
        // get data
        const pixelValues = new Uint8Array(4 * canvas.width * canvas.height)
        gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, pixelValues)
        return pixelValues
    }

    const unpackToFloat = (array: number[]) => {
        // convert packed floats to proper floats :
        // see http://stackoverflow.com/questions/9882716/packing-float-into-vec4-how-does-this-code-work
        const newArray = []
        const al = array.length
        for (let i = 0; i < al; i += 4) {
            newArray[(i / 4) >> 0] =
                array[i] / (256 * 256 * 256 * 256) +
                array[i + 1] / (256 * 256 * 256) +
                array[i + 2] / (256 * 256) +
                array[i + 3] / 256
        }
        return newArray
    }

    const normalizeFilterMatrix = (response: number[]) => {
        // normalize responses to lie within [0,1]
        const msize = response.length
        let max = 0
        let min = 1

        for (let i = 0; i < msize; i++) {
            max = response[i] > max ? response[i] : max
            min = response[i] < min ? response[i] : min
        }
        const dist = max - min

        if (dist === 0) {
            response = response.fill(1)
        } else {
            for (let i = 0; i < msize; i++) {
                response[i] = (response[i] - min) / dist
            }
        }

        return response
    }

    return {
        init,
        getRawResponses,
        getSobelResponses,
        getLBPResponses
    }
}
