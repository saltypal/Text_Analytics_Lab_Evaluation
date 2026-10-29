"""Ollama Qwen3 zero/few-shot classification; no fine-tuning or API key."""
import json
import os
import time
import subprocess
import shutil
import platform
from pathlib import Path
import requests
import numpy as np
from .preprocessing import LABELS

def install_and_load_ollama(config, output_dir):
    """Install Ollama on Colab/Linux, start it, pull Qwen, and warm-load it."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which('ollama') is None:
        if platform.system() != 'Linux':
            raise RuntimeError('Ollama installation is intended for Google Colab/Linux.')
        if shutil.which('zstd') is None:
            subprocess.run(['apt-get', 'update', '-qq'], check=True)
            subprocess.run(['apt-get', 'install', '-y', '-qq', 'zstd'], check=True)
        print('Installing Ollama using its official installer ...', flush=True)
        response = requests.get('https://ollama.com/install.sh', timeout=120)
        response.raise_for_status()
        installer = output_dir / 'install_ollama.sh'
        installer.write_text(response.text, encoding='utf-8')
        subprocess.run(['sh', str(installer)], check=True)
    start_ollama(config['ollama_url'], config['llm_model'], output_dir)
    response = requests.post(config['ollama_url']+'/api/generate',
                            json={'model': config['llm_model'], 'keep_alive': '10m'},
                            timeout=config['llm_timeout'])
    response.raise_for_status()
    print('DONE: Ollama installed, server ready, model loaded:', config['llm_model'], flush=True)

def classify_papers(sample, output_dir, config, system_prompt, example_messages=None, mode='Zero-shot LLM'):
    """One mode per call; zero-shot gets no demonstration messages."""
    schema = {'type': 'object', 'properties': {'domain': {'type': 'string', 'enum': LABELS}},
              'required': ['domain'], 'additionalProperties': False}
    examples = list(example_messages or [])
    records, predictions = [], []
    for i, row in enumerate(sample.itertuples(), 1):
        print(f'{mode}: {i}/{len(sample)} | {row.paper_id}', flush=True)
        messages = [{'role': 'system', 'content': system_prompt}] + examples + [{'role': 'user', 'content': row.text}]
        payload = {'model': config['llm_model'], 'messages': messages, 'stream': False,
                   'think': config['llm_think'], 'format': schema,
                   'options': {'temperature': config['llm_temperature'], 'seed': config['seed'],
                               'num_ctx': config['llm_context'], 'num_predict': config['llm_max_tokens']}}
        response = requests.post(config['ollama_url']+'/api/chat', json=payload, timeout=config['llm_timeout'])
        response.raise_for_status()
        raw = response.json()
        try:
            label = json.loads(raw['message']['content'])['domain']
            if label not in LABELS:
                label = 'INVALID'
        except (ValueError, KeyError, TypeError):
            label = 'INVALID'
        predictions.append(label)
        records.append({'System': mode, 'paper_id': row.paper_id, 'True': row.research_domain,
                        'Predicted': label, 'request': payload, 'response': raw})
        filename = mode.lower().replace(' ', '_').replace('-', '_') + '_evidence.json'
        (output_dir / filename).write_text(json.dumps(records, indent=2), encoding='utf-8')
        print('Predicted:', label, flush=True)
    return np.asarray(predictions)

def select_ten(test, seed=42):
    indices = []
    for label in LABELS:
        indices.extend(test[test.research_domain == label].sample(n=2, random_state=seed).index.tolist())
    return test.loc[indices].copy(), indices

def start_ollama(base_url, model_name, output_dir):
    try:
        requests.get(base_url+'/api/tags', timeout=3).raise_for_status()
    except requests.RequestException:
        with open(output_dir / 'ollama_server.log', 'a', encoding='utf-8') as log:
            subprocess.Popen(['ollama', 'serve'], stdout=log, stderr=subprocess.STDOUT)
        for _ in range(60):
            try:
                requests.get(base_url+'/api/tags', timeout=2).raise_for_status()
                break
            except requests.RequestException:
                time.sleep(1)
        else:
            raise RuntimeError('Ollama did not start. Inspect ollama_server.log.')
    print(f'Pulling {model_name}; download progress follows.', flush=True)
    subprocess.run(['ollama', 'pull', model_name], check=True, env={**os.environ, 'OLLAMA_HOST': base_url})
    response = requests.get(base_url+'/api/tags', timeout=30)
    response.raise_for_status()
    (output_dir/'ollama_models.json').write_text(json.dumps(response.json(), indent=2), encoding='utf-8')
    print('Ollama ready:', model_name, flush=True)
