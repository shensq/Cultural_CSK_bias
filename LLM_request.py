import argparse
import logging
import transformers
import datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='atomic_samples.txt')
    parser.add_argument('--max_new_tokens', type=int, default=30)
    parser.add_argument('--checkpoint', type=str, default="facebook/opt-iml-max-1.3b")
    args = parser.parse_args()

    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    logging.info(args)

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

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
    for i, sample in tqdm(enumerate(data[:])):
        inputs = tokenizer.encode(sample, return_tensors="pt").to("cuda")
        outputs = model.generate(inputs, max_new_tokens=args.max_new_tokens)
        results[i] = tokenizer.decode(outputs[0])
    with open("results/country_prediction/{}_{}.txt".format(args.checkpoint.split("/")[1], args.file[:-4]), 'w') as f:
        results = [r+'\n' for r in results]
        f.writelines(results)


if __name__ == "__main__":
    main()


