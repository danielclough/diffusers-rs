data-v2-1-cpu:check-venv
	git clone https://huggingface.co/lmz/rust-stable-diffusion-v2-1
	cp rust-stable-diffusion-v2-1/weights data -r
	# download vae.bin
	wget -O data/vae.bin https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/vae/diffusion_pytorch_model.bin
	# download unet.bin
	wget -O data/unet.bin https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/unet/diffusion_pytorch_model.bin
	# python to make safetensors
	echo "import torch\nfrom safetensors.torch import save_file\nmodel = torch.load('./vae.bin', map_location=torch.device('cpu'))\nsave_file(dict(model), './vae_v2.1.safetensors')\n\nmodel = torch.load('./unet.bin', map_location=torch.device('cpu'))\nsave_file(dict(model), './unet_v2.1.safetensors')\n" > data/vae-unet-build.py
	echo "import numpy as np\nimport torch\nfrom safetensors.torch import save_file\nmodel = torch.load('./pytorch_model.bin')\ntensors = {k: v.clone().detach() for k, v in model.items() if 'text_model' in k}\nsave_file(tensors, 'clip_v2.1.safetensors')" > data/clip-build.py
	# download pytorch_model.bin
	wget -O data/pytorch_model.bin https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin
	# create safetensors
	echo 'Create safetensors'
	cd data && python vae-unet-build.py && python clip-build.py
	cargo run --example stable-diffusion --features clap -- --prompt "A rusty robot holding a fire torch." --cpu all

data-v1-5:check-venv
	git clone https://huggingface.co/lmz/rust-stable-diffusion-v1-5
	cp rust-stable-diffusion-v1-5/weights data -r
	# download vae.bin
	wget -O data/vae.bin https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/vae/diffusion_pytorch_model.bin
	# download unet.bin
	wget -O data/unet.bin https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.bin
	# python to make safetensors
	echo "import torch\nfrom safetensors.torch import save_file\nmodel = torch.load('./vae.bin')\nsave_file(dict(model), './vae.safetensors')\n\nmodel = torch.load('./unet.bin')\nsave_file(dict(model), './unet.safetensors')\n" > data/vae-unet-build.py
	echo "import numpy as np\nimport torch\nfrom safetensors.torch import save_file\nmodel = torch.load('./pytorch_model.bin')\ntensors = {k: v.clone().detach() for k, v in model.items() if 'text_model' in k}\nsave_file(tensors, 'pytorch_model.safetensors')" > data/clip-build.py
	# download pytorch_model.bin
	wget -O data/pytorch_model.bin https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin
	# create safetensors
	echo '\tweights: '
	cd data && python vae-unet-build.py && python clip-build.py
	cargo run --example stable-diffusion --features clap -- --prompt "A rusty robot holding a fire torch." --cpu all  --sd-version v1-5

vocab:
	wget -O data/bpe_simple_vocab_16e6.txt.gz https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz
	cd data && gunzip bpe_simple_vocab_16e6.txt.gz

check-venv:
	if [ -d venv ]; then echo "venv is already setup"; else echo "***Seting up venv*** using 'make venv'"; make venv; fi

venv:
	python -m venv venv
	source venv/bin/activate
	pip install diffusers transformers accelerate scipy safetensors

purge:
	rm -fr data