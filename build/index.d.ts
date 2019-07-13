import { Model } from "./models";
export declare const Tracker: ({ searchWindow, scoreThreshold, stopOnConvergence, sharpenResponse, maxIterationsPerAnimFrame, weightPoints }: TrackerParams) => {
    init: (pmodel?: Model) => void;
    start: (element: HTMLVideoElement, box: number[]) => false | undefined;
    stop: () => void;
    getCurrentPosition: () => false | number[][];
};
