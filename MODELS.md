## Setting Up Models

To get started with training multimodal models, we need to pre-download the LLM weights and the vision-encoder weights. We support a variety of vision-encoders and LLMs as below:

### Vision-encoder weights
<table>
  <tr>
    <th>Vision Encoder</th>
    <th>HF&nbsp;Link</th>
  </tr>
  <tr>
    <td>InternViT&#8209;300M&#8209;448px&#8209;V2_5</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5">🤗 link</a></td>
  </tr>
</table>

### LLM weights
<table>
  <tr>
    <th>LLM</th>
    <th>HF&nbsp;Link</th>
  </tr>
  <tr>
    <td>Qwen2.5&#8209;0.5B</td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-0.5B">🤗 link</a></td>
  </tr>
  <tr>
    <td>Qwen2.5&#8209;1.5B</td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-1.5B">🤗 link</a></td>
  </tr>
  <tr>
    <td>Qwen2.5&#8209;3B</td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-3B">🤗 link</a></td>
  </tr>
  <tr>
    <td>Qwen2.5&#8209;7B</td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-7B">🤗 link</a></td>
  </tr>
</table>


To download models, you can use the below commands:
```bash
pip install -U huggingface_hub

huggingface-cli download --resume-download --local-dir-use-symlinks False <HF-DIR> --local-dir <LOCAL-DIR>

# For example, to download the Qwen2.5-0.5B model:
huggingface-cli download --resume-download --local-dir-use-symlinks False Qwen/Qwen2.5-0.5B --local-dir Qwen2.5-0.5B
```

If your compute nodes have no internet access, download the models to `OFFLINE_MODEL_DIR` (see `internvl_chat/.env.template`).
