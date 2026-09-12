import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

disable_torch_init()

model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-AV'
model, processor, tokenizer = model_init(model_path, device_map={"": "cuda"})

output = mm_infer(
    processor['image']('assets/sora.png'),
    'What is in this image?',
    model=model, tokenizer=tokenizer,
    do_sample=False, modal='image'
)
print(output)