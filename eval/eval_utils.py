from estimator.estimator_llm import LLMEstimator


def set_function_from_iterrow(func):
    def wrapper(dataset):
        dataset['score'] = dataset.apply(func, axis=1)
        return dataset

    return wrapper


def set_ranking_function(params):
    evaluator = LLMEstimator(params)
    evaluator.init_chain(params.label_schema)
    evaluator.mode = 'score'
    def wrapper(dataset):
        generation_dataset = dataset.copy()
        print(generation_dataset.head(1))
        # generation_dataset['text'] = '###User input:\n' + generation_dataset['text'] + '\n####model prediction:\n' + generation_dataset['prediction']
        generation_dataset['text'] = '###用户输入:\n' + generation_dataset['text'] + '\n####模型预测:\n' + generation_dataset['prediction']

        generation_dataset = evaluator.apply_dataframe(generation_dataset)
        generation_dataset = generation_dataset[generation_dataset['score'] != 'Discarded']
        generation_dataset.score = generation_dataset.score.astype(int)
        dataset.score = generation_dataset.score
        return dataset
    return wrapper
