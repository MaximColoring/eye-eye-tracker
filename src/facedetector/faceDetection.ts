import mosse from "mosse"
import jsfeat from "jsfeat"
import { frontalface } from "./classifier_frontalface"

import { Model } from "../models"

type FaceDetectionParams = {
    workSize?: number
    minScale?: number
    scaleFactor?: number
    useCanny?: boolean
    edgesDensity?: number
    equalizeHistogram?: boolean
    min_neighbors?: number
    confidenceThreshold?: number
}

export type FaceDetector = ReturnType<typeof faceDetector>

export const faceDetector = (model: Model, params: FaceDetectionParams) => {
    let element: HTMLCanvasElement

    let mossef_lefteye: any
    let mossef_righteye: any
    let mossef_nose: any

    const right_eye_position = [0.0, 0.0]
    const left_eye_position = [0.0, 0.0]
    const nose_position = [0.0, 0.0]

    const mosseExists =
        mosse.mosseFilter &&
        mosse.filters.left_eye_filter &&
        mosse.filters.right_eye_filter &&
        mosse.filters.nose_filter

    if (model.hints && mosseExists) {
        mossef_lefteye = new mosse.mosseFilter()
        mossef_lefteye.load(mosse.filters.left_eye_filter)
        mossef_righteye = new mosse.mosseFilter()
        mossef_righteye.load(mosse.filters.right_eye_filter)
        mossef_nose = new mosse.mosseFilter()
        mossef_nose.load(mosse.filters.nose_filter)
    } else {
        console.log("MOSSE filters not found, using rough approximation for initialization.")
    }

    // load mean shape
    const meanShape = model.shapeModel.meanShape
    const numPatches = model.patchModel.numPatches

    // get max and mins, width and height of meanshape
    let msymax = 0
    let msxmin = 1000000
    let msymin = 1000000
    for (let i = 0; i < numPatches; i++) {
        if (meanShape[i][0] < msxmin) msxmin = meanShape[i][0]
        if (meanShape[i][1] < msymin) msymin = meanShape[i][1]
        if (meanShape[i][1] > msymax) msymax = meanShape[i][1]
    }
    const msmodelheight = msymax - msymin

    const jf = jsfeat_face(params)

    const init = (video: HTMLCanvasElement) => {
        element = video

        jf.init(element)
    }

    const getBoundingBox = (box?: number[]) =>
        box ? { x: box[0], y: box[1], width: box[2], height: box[3] } : jf.findFace()

    const getFinegrainedPosition = (candidate: Box) => {
        let translateX
        let translateY
        let scaling
        let rotation
        const x = candidate.x
        const y = candidate.y
        const w = candidate.width
        const h = candidate.height

        // var debugCC = document.getElementById('overlay2').getContext('2d')
        if (model.hints && mosseExists) {
            const noseFilterWidth = (w * 4.5) / 10
            const eyeFilterWidth = (w * 6) / 10

            // detect position of eyes and nose via mosse filter
            const nose_result = mossef_nose.track(
                element,
                Math.round(x + w / 2 - noseFilterWidth / 2),
                Math.round(y + h * (5 / 8) - noseFilterWidth / 2),
                noseFilterWidth,
                noseFilterWidth,
                false
            )
            const right_result = mossef_righteye.track(
                element,
                Math.round(x + (w * 3) / 4 - eyeFilterWidth / 2),
                Math.round(y + h * (2 / 5) - eyeFilterWidth / 2),
                eyeFilterWidth,
                eyeFilterWidth,
                false
            )
            const left_result = mossef_lefteye.track(
                element,
                Math.round(x + w / 4 - eyeFilterWidth / 2),
                Math.round(y + h * (2 / 5) - eyeFilterWidth / 2),
                eyeFilterWidth,
                eyeFilterWidth,
                false
            )
            right_eye_position[0] = Math.round(x + (w * 3) / 4 - eyeFilterWidth / 2) + right_result[0]
            right_eye_position[1] = Math.round(y + h * (2 / 5) - eyeFilterWidth / 2) + right_result[1]
            left_eye_position[0] = Math.round(x + w / 4 - eyeFilterWidth / 2) + left_result[0]
            left_eye_position[1] = Math.round(y + h * (2 / 5) - eyeFilterWidth / 2) + left_result[1]
            nose_position[0] = Math.round(x + w / 2 - noseFilterWidth / 2) + nose_result[0]
            nose_position[1] = Math.round(y + h * (5 / 8) - noseFilterWidth / 2) + nose_result[1]

            // drawDetection(debugCC, candidate, [left_eye_position, right_eye_positions, nose_position]);

            // get eye and nose positions of model
            const lep = model.hints.leftEye
            const rep = model.hints.rightEye
            const mep = model.hints.nose

            // get scaling, rotation, etc. via procrustes analysis
            const procrustes_params = procrustes(
                [left_eye_position, right_eye_position, nose_position],
                [lep, rep, mep]
            )
            translateX = procrustes_params[0]
            translateY = procrustes_params[1]
            scaling = procrustes_params[2]
            rotation = procrustes_params[3]

            // drawFacialPoints(debugCC, [lep, rep, mep], procrustes_params);
        } else {
            // drawBoundingBox(debugCC, [x,y,w,h]);
            scaling = w / msmodelheight
            rotation = 0
            translateX = x - msxmin * scaling + 0.1 * w
            translateY = y - msymin * scaling + 0.25 * h
        }

        return [scaling, rotation, translateX, translateY]
    }

    // get initial starting point for model
    const getInitialPosition = (box?: number[]) => {
        const bBox = getBoundingBox(box)
        if (!bBox) return false
        const fPos = getFinegrainedPosition(bBox)
        return fPos
    }

    // procrustes analysis
    const procrustes = (template: number[][], shape: number[][]) => {
        // assume template and shape is a vector of x,y-coordinates
        // i.e. template = [[x1,y1], [x2,y2], [x3,y3]];
        const templateClone = []
        const shapeClone = []
        for (let i = 0; i < template.length; i++) {
            templateClone[i] = [template[i][0], template[i][1]]
        }
        for (let i = 0; i < shape.length; i++) {
            shapeClone[i] = [shape[i][0], shape[i][1]]
        }
        shape = shapeClone
        template = templateClone

        // calculate translation
        const templateMean = [0.0, 0.0]
        for (const t of template) {
            templateMean[0] += t[0]
            templateMean[1] += t[1]
        }
        templateMean[0] /= template.length
        templateMean[1] /= template.length

        const shapeMean = [0.0, 0.0]
        for (const s of shape) {
            shapeMean[0] += s[0]
            shapeMean[1] += s[1]
        }
        shapeMean[0] /= shape.length
        shapeMean[1] /= shape.length

        let translationX = templateMean[0] - shapeMean[0]
        let translationY = templateMean[1] - shapeMean[1]

        // centralize
        for (const s of shape) {
            s[0] -= shapeMean[0]
            s[1] -= shapeMean[1]
        }
        for (const t of template) {
            t[0] -= templateMean[0]
            t[1] -= templateMean[1]
        }

        // scaling

        let scaleS = 0.0
        for (const s of shape) {
            scaleS += s[0] * s[0]
            scaleS += s[1] * s[1]
        }
        scaleS = Math.sqrt(scaleS / shape.length)

        let scaleT = 0.0
        for (const t of template) {
            scaleT += t[0] * t[0]
            scaleT += t[1] * t[1]
        }
        scaleT = Math.sqrt(scaleT / template.length)

        const scaling = scaleT / scaleS

        for (const s of shape) {
            s[0] *= scaling
            s[1] *= scaling
        }

        // rotation

        let top = 0.0
        let bottom = 0.0
        for (let i = 0; i < shape.length; i++) {
            top += shape[i][0] * template[i][1] - shape[i][1] * template[i][0]
            bottom += shape[i][0] * template[i][0] + shape[i][1] * template[i][1]
        }
        const rotation = Math.atan(top / bottom)

        translationX +=
            shapeMean[0] - scaling * Math.cos(-rotation) * shapeMean[0] - scaling * shapeMean[1] * Math.sin(-rotation)
        translationY +=
            shapeMean[1] + scaling * Math.sin(-rotation) * shapeMean[0] - scaling * shapeMean[1] * Math.cos(-rotation)

        return [translationX, translationY, scaling, rotation]
    }

    return { getInitialPosition, init }
}

// simple wrapper for jsfeat face detector that can run as a webworker
const jsfeat_face = ({
    workSize = 200,
    minScale = 2,
    scaleFactor = 1.15,
    useCanny = false,
    edgesDensity = 0.13,
    equalizeHistogram = false,
    min_neighbors = 2,
    confidenceThreshold = 106.1
}: FaceDetectionParams) => {
    const work_canvas = document.createElement("canvas")!
    const work_ctx = work_canvas.getContext("2d")!

    let videoWidth: number
    let videoHeight: number
    let scale: number
    let video: HTMLCanvasElement
    let w: number
    let h: number
    let img_u8: Int32Array
    let edg: number[][]
    let ii_sum: Int32Array
    let ii_sqsum: Int32Array
    let ii_tilted: Int32Array
    let ii_canny: Int32Array
    let classifier: any

    const init = (element: HTMLCanvasElement) => {
        video = element
        videoWidth = video.width
        videoHeight = video.height

        // scale down canvas we do detection on (to reduce noisy detections)
        scale = Math.min(workSize! / videoWidth, workSize! / videoHeight)
        w = (videoWidth * scale) | 0
        h = (videoHeight * scale) | 0

        work_canvas.height = h
        work_canvas.width = w

        img_u8 = new jsfeat.matrix_t(w, h, jsfeat.U8_t | jsfeat.C1_t)
        edg = new jsfeat.matrix_t(w, h, jsfeat.U8_t | jsfeat.C1_t)
        ii_sum = new Int32Array((w + 1) * (h + 1))
        ii_sqsum = new Int32Array((w + 1) * (h + 1))
        ii_tilted = new Int32Array((w + 1) * (h + 1))
        ii_canny = new Int32Array((w + 1) * (h + 1))
        classifier = frontalface
    }

    const findFace = () => {
        work_ctx.drawImage(video, 0, 0, work_canvas.width, work_canvas.height)
        const imageData = work_ctx.getImageData(0, 0, work_canvas.width, work_canvas.height)

        jsfeat.imgproc.grayscale(imageData.data, work_canvas.width, work_canvas.height, img_u8)

        // possible params
        if (equalizeHistogram) {
            jsfeat.imgproc.equalize_histogram(img_u8, img_u8)
        }
        // jsfeat.imgproc.gaussian_blur(img_u8, img_u8, 3);

        jsfeat.imgproc.compute_integral_image(img_u8, ii_sum, ii_sqsum, classifier.tilted ? ii_tilted : null)

        if (useCanny) {
            jsfeat.imgproc.canny(img_u8, edg, 10, 50)
            jsfeat.imgproc.compute_integral_image(edg, ii_canny, null, null)
        }

        jsfeat.haar.edgesDensity = edgesDensity
        let rects = jsfeat.haar.detect_multi_scale(
            ii_sum,
            ii_sqsum,
            ii_tilted,
            useCanny ? ii_canny : null,
            (img_u8 as any).cols,
            (img_u8 as any).rows,
            classifier,
            scaleFactor,
            minScale
        )
        rects = jsfeat.haar.group_rectangles(rects, min_neighbors)

        for (let i = rects.length - 1; i >= 0; i--) {
            if (rects[i].confidence < confidenceThreshold!) {
                rects.splice(i, 1)
            }
        }

        const rl = rects.length
        if (rl === 0) {
            return
        } else {
            let best = rects[0]
            for (let i = 1; i < rl; i++) {
                if (rects[i].neighbors > best.neighbors) {
                    best = rects[i]
                } else if (rects[i].neighbors === best.neighbors) {
                    // if (rects[i].width > best.width) best = rects[i]; // use biggest rect
                    if (rects[i].confidence > best.confidence) best = rects[i] // use most confident rect
                }
            }

            const sc = videoWidth / (img_u8 as any).cols
            best.x = (best.x * sc) | 0
            best.y = (best.y * sc) | 0
            best.width = (best.width * sc) | 0
            best.height = (best.height * sc) | 0

            return best as Box
        }
    }

    return { findFace, init }
}
