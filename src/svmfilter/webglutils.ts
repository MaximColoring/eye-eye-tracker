export const setupWebGL = (canvas: HTMLCanvasElement, attrs?: WebGLContextAttributes) => {
    if (!(window as any).WebGLRenderingContext) {
        // showLink(GET_A_WEBGL_BROWSER);
        return null
    }
    try {
        return canvas.getContext("webgl", attrs)
    } catch (e) {
        try {
            return canvas.getContext("experimental-webgl", attrs)
        } catch (e2) {
            return null
        }
    }
}

/**
 * Gets a WebGL context.
 * makes its backing store the size it is displayed.
 */
export const getWebGLContext = (canvas: HTMLCanvasElement) => setupWebGL(canvas)

export const loadShader = (
    gl: WebGLRenderingContext,
    shaderSource: string,
    shaderType: number,
    optErrorCallback?: (err: string) => any
) => {
    const errFn = optErrorCallback || console.error
    // Create the shader object
    const shader = gl.createShader(shaderType)
    if (!shader) return null
    // Load the shader source
    gl.shaderSource(shader, shaderSource)

    // Compile the shader
    gl.compileShader(shader)

    // Check the compile status
    const compiled = gl.getShaderParameter(shader, gl.COMPILE_STATUS)
    if (!compiled) {
        // Something went wrong during compilation; get the error
        const lastError = gl.getShaderInfoLog(shader)
        errFn("*** Error compiling shader '" + shader + "':" + lastError)
        gl.deleteShader(shader)
        return null
    }

    return shader
}

export const createProgram = (
    gl: WebGLRenderingContext,
    shaders: WebGLShader[],
    attrs?: string[],
    locations?: number[]
) => {
    const program = gl.createProgram()
    if (!program) return null
    for (const shader of shaders) {
        gl.attachShader(program, shader)
    }
    if (attrs) {
        for (let i = 0; i < attrs.length; ++i) {
            gl.bindAttribLocation(program, locations ? locations[i] : i, attrs[i])
        }
    }
    gl.linkProgram(program)

    // Check the link status
    const linked = gl.getProgramParameter(program, gl.LINK_STATUS)
    if (!linked) {
        // something went wrong with the link
        const lastError = gl.getProgramInfoLog(program)
        console.error("Error in program linking:" + lastError)

        gl.deleteProgram(program)
        return null
    }
    return program
}
