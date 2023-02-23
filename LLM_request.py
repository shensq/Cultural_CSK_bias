import argparse
import pickle
import logging
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='atomic_samples.txt')
    parser.add_argument('--max_tokens', type=int, default=256)
    parser.add_argement('--checkpoint', type=str, default="facebook/opt-iml-max-1.3b")
    args = parser.parse_args()

    logging.info(args)

    logging.info("Loading Dataset.")
    folder = "data/country_prediction/"
    with open(folder + args.file, 'r') as f:
        data = f.readlines()
    data = [d.replace('\n', '') for d in data]
    logging.info("Dataset loaded")

    logging.info("Loading model {}".format(args.checkpoint))
    checkpoint = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

    results = [None] * len(data)
    for i, sample in tqdm(enumerate(data)):
        inputs = tokenizer.encode(sample, return_tensors="pt").to("cuda")
        outputs = model.generate(inputs, max_new_tokens=20)
        results[i] = tokenizer.decode(outputs[0])
        with open("gpt3/country_prediction/{}_{}.pkl".format(args.model.split("/"[1]), args.file[:-4]), 'w') as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    main()


