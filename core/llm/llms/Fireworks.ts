import { LLMOptions, ModelProvider } from "../../index.js";
import OpenAI from "./OpenAI.js";

class Fireworks extends OpenAI {
  static providerName: ModelProvider = "fireworks";
  static defaultOptions: Partial<LLMOptions> = {
    apiBase: "https://api.fireworks.ai/inference/v1",
  };

  private static modelConversion: { [key: string]: string } = {
    "starcoder-7b": "accounts/fireworks/models/starcoder-7b",
    "llama-v3p1-405b-instruct":
      "accounts/fireworks/models/llama-v3p1-405b-instruct",
    "llama-v3p1-70b-instruct":
      "accounts/fireworks/models/llama-v3p1-70b-instruct",
    "llama-v3p1-8b-instruct":
      "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "llama-v3-70b-instruct": "accounts/fireworks/models/llama-v3-70b-instruct",
    "mixtral-8x22b-instruct":
      "accounts/fireworks/models/mixtral-8x22b-instruct",
    "mixtral-8x7b-instruct": "accounts/fireworks/models/mixtral-8x7b-instruct",
    "firefunction-v2": "accounts/fireworks/models/firefunction-v2",
    "firellava-13b": "accounts/fireworks/models/firellava-13b",
    "chronos-hermes-13b-v2": "accounts/fireworks/models/chronos-hermes-13b-v2",
    "codegemma-2b": "accounts/fireworks/models/codegemma-2b",
    "codegemma-7b": "accounts/fireworks/models/codegemma-7b",
    "code-llama-13b": "accounts/fireworks/models/code-llama-13b",
    "code-llama-13b-instruct":
      "accounts/fireworks/models/code-llama-13b-instruct",
    "code-llama-13b-python": "accounts/fireworks/models/code-llama-13b-python",
    "code-llama-34b": "accounts/fireworks/models/code-llama-34b",
    "code-llama-34b-instruct":
      "accounts/fireworks/models/code-llama-34b-instruct",
    "code-llama-34b-python": "accounts/fireworks/models/code-llama-34b-python",
    "code-llama-70b": "accounts/fireworks/models/code-llama-70b",
    "code-llama-70b-instruct":
      "accounts/fireworks/models/code-llama-70b-instruct",
    "code-llama-70b-python": "accounts/fireworks/models/code-llama-70b-python",
    "code-llama-7b": "accounts/fireworks/models/code-llama-7b",
    "code-llama-7b-instruct":
      "accounts/fireworks/models/code-llama-7b-instruct",
    "code-llama-7b-python": "accounts/fireworks/models/code-llama-7b-python",
    "code-qwen-1p5-7b": "accounts/fireworks/models/code-qwen-1p5-7b",
    "deepseek-coder-1b-base":
      "accounts/fireworks/models/deepseek-coder-1b-base",
    "deepseek-coder-33b-instruct":
      "accounts/fireworks/models/deepseek-coder-33b-instruct",
    "deepseek-coder-6.7b-base":
      "accounts/fireworks/models/deepseek-coder-7b-base",
    "deepseek-coder-7b-base-v1p5":
      "accounts/fireworks/models/deepseek-coder-7b-base-v1p5",
    "deepseek-coder-v2-lite-base":
      "accounts/fireworks/models/deepseek-coder-v2-lite-base",
    "deepseek-v2p5": "accounts/fireworks/models/deepseek-v2p5",
    "dolphin-2-9-2-qwen2-72b":
      "accounts/fireworks/models/dolphin-2-9-2-qwen2-72b",
    "dolphin-2p6-mixtral-8x7b":
      "accounts/fireworks/models/dolphin-2p6-mixtral-8x7b",
    "elyza-japanese-llama-2-7b-fast-instruct":
      "accounts/fireworks/models/elyza-japanese-llama-2-7b-fast-instruct",
    "gemma2-9b-it": "accounts/fireworks/models/gemma2-9b-it",
    "gemma-7b": "accounts/fireworks/models/gemma-7b",
    "hermes-2-pro-mistral-7b":
      "accounts/fireworks/models/hermes-2-pro-mistral-7b",
    "japanese-stablelm-instruct-beta-70b":
      "accounts/stability/models/japanese-stablelm-instruct-beta-70b",
    "japanese-stablelm-instruct-gamma-7b":
      "accounts/stability/models/japanese-stablelm-instruct-gamma-7b",
    "llama-guard-2-8b": "accounts/fireworks/models/llama-guard-2-8b",
    "llamaguard-7b": "accounts/fireworks/models/llamaguard-7b",
    "llama-v2-7b": "accounts/fireworks/models/llama-v2-7b",
    "llama-v3-70b-instruct-hf":
      "accounts/fireworks/models/llama-v3-70b-instruct-hf",
    "llama-v3-8b-hf": "accounts/fireworks/models/llama-v3-8b-hf",
    "llama-v3-8b-instruct": "accounts/fireworks/models/llama-v3-8b-instruct",
    "llama-v3-8b-instruct-hf":
      "accounts/fireworks/models/llama-v3-8b-instruct-hf",
    "llava-yi-34b": "accounts/fireworks/models/llava-yi-34b",
    "mistral-7b": "accounts/fireworks/models/mistral-7b",
    "mistral-7b-v0p2": "accounts/fireworks/models/mistral-7b-v0p2",
    "mistral-nemo-base-2407":
      "accounts/fireworks/models/mistral-nemo-base-2407",
    "mistral-nemo-instruct-2407":
      "accounts/fireworks/models/mistral-nemo-instruct-2407",
    "mixtral-8x22b": "accounts/fireworks/models/mixtral-8x22b",
    "mixtral-8x22b-hf": "accounts/fireworks/models/mixtral-8x22b-hf",
    "mixtral-8x7b-instruct-hf":
      "accounts/fireworks/models/mixtral-8x7b-instruct-hf",
    "mythomax-l2-13b": "accounts/fireworks/models/mythomax-l2-13b",
    "nous-capybara-7b-v1p9": "accounts/fireworks/models/nous-capybara-7b-v1p9",
    "nous-hermes-2-mixtral-8x7b-dpo":
      "accounts/fireworks/models/nous-hermes-2-mixtral-8x7b-dpo",
    "nous-hermes-2-mixtral-8x7b-dpo-fp8":
      "accounts/fireworks/models/nous-hermes-2-mixtral-8x7b-dpo-fp8",
    "nous-hermes-2-yi-34b": "accounts/fireworks/models/nous-hermes-2-yi-34b",
    "nous-hermes-llama2-13b":
      "accounts/fireworks/models/nous-hermes-llama2-13b",
    "nous-hermes-llama2-70b":
      "accounts/fireworks/models/nous-hermes-llama2-70b",
    "nous-hermes-llama2-7b": "accounts/fireworks/models/nous-hermes-llama2-7b",
    "openchat-3p5-0106-7b": "accounts/fireworks/models/openchat-3p5-0106-7b",
    "openhermes-2-mistral-7b":
      "accounts/fireworks/models/openhermes-2-mistral-7b",
    "openhermes-2p5-mistral-7b":
      "accounts/fireworks/models/openhermes-2p5-mistral-7b",
    "openorca-7b": "accounts/fireworks/models/openorca-7b",
    "phi-2-3b": "accounts/fireworks/models/phi-2-3b",
    "phind-code-llama-34b-python-v1":
      "accounts/fireworks/models/phind-code-llama-34b-python-v1",
    "phind-code-llama-34b-v1":
      "accounts/fireworks/models/phind-code-llama-34b-v1",
    "phind-code-llama-34b-v2":
      "accounts/fireworks/models/phind-code-llama-34b-v2",
    "pythia-12b": "accounts/fireworks/models/pythia-12b",
    "qwen1p5-72b-chat": "accounts/fireworks/models/qwen1p5-72b-chat",
    "snorkel-mistral-7b-pairrm-dpo":
      "accounts/fireworks/models/snorkel-mistral-7b-pairrm-dpo",
    "starcoder-16b": "accounts/fireworks/models/starcoder-16b",
    "starcoder2-15b": "accounts/fireworks/models/starcoder2-15b",
    "starcoder2-3b": "accounts/fireworks/models/starcoder2-3b",
    "starcoder2-7b": "accounts/fireworks/models/starcoder2-7b",
    "toppy-m-7b": "accounts/fireworks/models/toppy-m-7b",
    "yi-34b": "accounts/fireworks/models/yi-34b",
    "yi-34b-200k-capybara": "accounts/fireworks/models/yi-34b-200k-capybara",
    "yi-34b-chat": "accounts/fireworks/models/yi-34b-chat",
    "yi-6b": "accounts/fireworks/models/yi-6b",
    "zephyr-7b-beta": "accounts/fireworks/models/zephyr-7b-beta",
  };

  protected _convertModelName(model: string): string {
    return Fireworks.modelConversion[model] ?? model;
  }
}

export default Fireworks;
