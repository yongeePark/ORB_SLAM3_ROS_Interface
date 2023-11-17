import tensorrt as trt
 
onnx_file_name = 'my_model.onnx'
tensorrt_file_name = 'my_model.trt'
fp_16_mode = True
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
 
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(EXPLICIT_BATCH)
parser = trt.OnnxParser(network, TRT_LOGGER)
 
# builder.max_workspace_size = (1 << 30)
builder.fp16_mode = fp16_mode
 
with open(onnx_file_name, 'rb') as model:
    if not parser.parse(model.read()):
        for error in range(parser.num_errors):
            print (parser.get_error(error))
 
engine = builder.build_cuda_engine(network)
buf = engine.serialize()
with open(tensorrt_file_name, 'wb') as f:
    f.write(buf)
