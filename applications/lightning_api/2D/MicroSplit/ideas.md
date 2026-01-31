1) Too many concepts before first success
    To get any result, a user must implicitly understand many things:
	    data types (“tiff” vs “array”), axes strings, patch/tile/grid/multiscale, NM + N2V pretraining, etc

2) Long time for any result
    Something like (task/pipeline).estimate() returns rough time/memory ?
    Preview mode: ability to run train/prediction on a small crop quickly.
    (task/pipeline).summary() maybe show main config and/or summary of the run?

3) Configuration is huge
    progressive disclosure config: basic parameters visible, full list ->  .advanced(...)
    several presets?


4) Noise models are a pain ...


5) Obviously all parameters should be cross-validated
    Fail fast with friendly messages.


6) Collapse correlated parameters into objects(partially done)
    e.g.: encoder/decoder conv strides + filters + dropout + nonlinearity → ModelConfig

7) Return Prediction objects with metadata and view/save/... methods.