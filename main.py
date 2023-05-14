from pyllamacpp.model import Model

print('---start---\n')
model = Model('./gpt4all-lora-quantized-ggjt.bin')
promptText = 'Please introduce yourself.' # ここに文章を入力
for token in model.generate(prompt=promptText):
    print(token, end='', flush=True)
print('\n---end---')