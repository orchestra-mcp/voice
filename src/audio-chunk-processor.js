class AudioChunkProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = [];
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      // Copy channel data (can't transfer SharedArrayBuffer across threads safely here)
      this.port.postMessage(new Float32Array(input[0]));
    }
    return true; // keep processor alive
  }
}

registerProcessor('audio-chunk-processor', AudioChunkProcessor);
