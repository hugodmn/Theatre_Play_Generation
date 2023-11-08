import torch 
import onnx 
import onnxruntime as ort 
from optimum.onnxruntime import ORTModelForAudioClassification
from onnxruntime import SessionOptions
#from onnx_tf.backend import prepare
import numpy as np 
from models.model import LLM, Config
import torch
import json

def loading_model(model, weights_path):
   
        #Load json file in a dict
        with open(('/').join(weights_path.split('/')[:-1])+'/config.json', 'r') as f:
            config_dict = json.load(f)
        config = Config(**config_dict)

        inference = model(config)
        inference.eval()

        return inference


def quantize_onnx_model(onnx_model_path, quantized_model_path):
    print("Starting quantization...")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QUInt8)

    print(f"Quantized model saved to: {quantized_model_path}")


def saving_onnx(torch_model, saved_name, quantization = False, export = True):
    

        # def generate(self, idx, max_new_tokens):
        # # idx is (B, T) array of indices in the current context
        # for _ in range(max_new_tokens):
        #     # crop idx to the last block_size tokens
        #     idx_cond = idx[:, -self.block_size:]
        #     # get the predictions
        #     logits = self(idx_cond)
        #     # focus only on the last time step
        #     logits = logits[:, -1, :] # becomes (B, C)
        #     # apply softmax to get probabilities
        #     probs = F.softmax(logits, dim=-1) # (B, C)
        #     # sample from the distribution
        #     idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        #     # append sampled index to the running sequence
        #     idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

    # Input to the model
    #torch_in = torch.randn( 1, torch_model.config.block_size, requires_grad=True, dtype=torch.int32)
    torch_in = torch.randint(1, torch_model.config.vocab_size, (1, torch_model.config.block_size), dtype=torch.int32)
    torch_out = torch_model(torch_in)
    # dynamic_axes = {
    #     'input' : {1: 'audio_len'},
    #  }
    # Export the model
    if export :
        torch.onnx.export(torch_model,               # model being run
                        torch_in,                         # model input (or a tuple for multiple inputs)
                        saved_name,   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=9,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'],) # the model's output names
                        #dynamic_axes=dynamic_axes)    # variable length axes

        if quantization :
            quantize_onnx_model("ONNX_saved/"+saved_name+".onnx", "ONNX_saved/"+saved_name+".quant.onnx")

    return torch_in, torch_out

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def onnx_test(onnx_path, torch_in, torch_out):

    sess_options = SessionOptions()
    sess = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": to_numpy(torch_in)})[0]


    np.testing.assert_allclose(to_numpy(torch_out)[0], onnx_out[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

# def export_to_tf(onnx_path, tf_path):
#     onnx_model = onnx.load(onnx_path)
#     tf_rep = prepare(onnx_model)
#     tf_rep.export_graph(tf_path)
    

if __name__ == "__main__":
    weights_file_name = "char_level"
    quantization = False

    model_inf = loading_model(LLM,'models/char_tokens/model_[T:char_level].pt' )
    torch_in, torch_out = saving_onnx(model_inf, 'models/char_tokens/'+weights_file_name+'.onnx', quantization, export = True)
    onnx_test(onnx_path='models/char_tokens/'+weights_file_name+'.onnx',torch_in = torch_in, torch_out = torch_out)

    #export_to_tf('models/char_tokens/'+weights_file_name+'.onnx', 'models/char_tokens/'+weights_file_name+'.pb')

    # model_path = "ONNX_saved/w2v2-xlsr-6-norm-scripted.onnx"
    # sess_options = SessionOptions()
    # sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
    # onnx_out = sess.run(None, {"input": to_numpy(torch_in)})[0]
    # print(torch_out)
    # print(onnx_out)