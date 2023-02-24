import json
import random
import logging


def load_atomic():
    with open("data/top1_set.json", 'r') as f:
        lines = f.readlines()
        lines = [json.loads(l) for l in lines]

    verbalizer = {"xAttr": "If a person{}, he is seen as {}.",
                  "xEffect": "If a person{}, he {} as a result.",
                  "xIntent": "A person{} in order {}.",
                  "xNeed": "To{}, one needs {}.",
                  "xReact": "If a person{}, he will feel {}.",
                  "xWant": "If a person{}, he wants {}.",
                  "HinderedBy": "If a person wants to {}, that can be hindered by {}."}

    for i, l in enumerate(lines):
        l["head"] = l["head"].replace("PersonX", "")
        l["head"] = l["head"].replace("PersonY", "others")
        l = verbalizer[l["relation"]].format(l["head"], l["tail"])
        lines[i] = l
    return lines


def generate_cultural_group(dataset, dataset_name):
    cultural_group = ["United States", "India"]
    for cg in cultural_group:
        lines = []
        for d in dataset:
            lines.append("{} Is this true or false in {}?\n".format(d, cg))
        with open("data/assertions_{}_{}.txt".format(dataset_name, cg), 'w') as f:
            f.writelines(lines)
    return


def build_prompt(main_prompts, additional_prompt, data, ):
    num_templates = len(main_prompts)
    num_samples = len(data)

    samples = [""] * num_samples * num_templates
    for i, a in enumerate(data):
        a = a.replace("\n", ' ')
        for j in range(num_templates):
            samples[i * num_templates + j] = a + main_prompts[j].replace("?", ', ') + additional_prompt
    return samples


def get_samples(dataset, main_templates, additional_prompt, num_samples=-1):
    with open('country_prediction/{}_assertion.txt'.format(dataset), 'r') as f:
        data = f.readlines()
        if num_samples == -1:
            num_samples = len(data)
        random.seed(42)
        data = random.choices(data, k=num_samples)
        if dataset == "atomic":
            data = [d.replace("PersonY", "others") for d in data]  # a quick fix to PersonY

    samples = build_prompt(main_templates, additional_prompt, data, )

    with open("country_prediction/{}_samples.txt".format(dataset), 'w') as f:
        f.writelines("\n".join(samples))
    return samples


def main():
    atomic_templates = ["In which country did this event happen?",
                        "What nation was the site of this event?",
                        "What nation was the location of this occurrence?",
                        "The event is more likely to happen in which country?",
                        "Which nation is most probable to experience the occurrence?",
                        "Which country is the event most probable to occur in?"]

    genericskb_templates = ["The previous sentence is more likely to be true in which country?",
                            "What nation is most likely to have the preceding statement be accurate?",
                            "What country is more likely to have the previous statement be true?",
                            "The previous sentence is describing things happens in which country? ",
                            "What nation is being alluded to in the prior statement?",
                            "Which nation is the preceding statement referring to in regards to the events taking place?"]

    additional_prompt = "the United States, China, India, or Germany?"
    logging.info("Processing atomic!")
    samples = get_samples("atomic", atomic_templates, additional_prompt, num_samples=1500)

    logging.info("Processing GenericsKB!")
    samples = get_samples("genericskb", genericskb_templates, additional_prompt, num_samples=1500)


if __name__ == "__main__":
    main()
