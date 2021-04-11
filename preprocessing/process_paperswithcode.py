"""
Extract triples from paperswithcode released data
Author: Jiaying Lu
Create Date: Apr 11, 2021
"""

import json


def extract_areas(inpath: str, outpath: str):
    areas = set()
    with open(inpath) as fopen:
        data = json.load(fopen)
    for table in data:
        for area in table['categories']:
            areas.add(area.lower())
    with open(outpath, 'w') as fwrite:
        for area in areas:
            fwrite.write('%s\tIsA\tresearch area\n' % (area))


def extract_tasks(inpath: str, outpath: str):
    subtask_task_set = set()
    task_area_set = set()
    with open(inpath) as fopen:
        data = json.load(fopen)
    for table in data:
        task = table['task'].lower()
        for area in table['categories']:
            area = area.lower()
            if len(area) > 0 and len(task) > 0:
                task_area_set.add((task, area))
        for subtask_d in table['subtasks']:
            subtask = subtask_d['task'].lower()
            if len(subtask) > 0 and len(task) > 0:
                subtask_task_set.add((subtask, task))
            for area in subtask_d['categories']:
                area = area.lower()
                if len(area) > 0 and len(subtask) > 0:
                    task_area_set.add((subtask, area))
    with open(outpath, 'w') as fwrite:
        for task, area in task_area_set:
            fwrite.write('%s\tIsA\t%s\n' % (task, area))
        for subtask, task in subtask_task_set:
            fwrite.write('%s\tIsA\t%s\n' % (subtask, task))


def extract_metrics(inpath: str, outpath: str):
    metric_set = set()
    metric_task_set = set()
    metric_dataset_set = set()
    with open(inpath) as fopen:
        data = json.load(fopen)
    for table in data:
        task = table['task'].lower()
        for datasets_d in table['datasets']:
            dataset = datasets_d['dataset'].lower()
            for metric in datasets_d['sota']['metrics']:
                metric = metric.lower()
                if len(metric) > 0:
                    metric_set.add(metric)
                if len(metric) > 0 and len(task) > 0:
                    metric_task_set.add((metric, task))
                if len(metric) > 0 and len(dataset) > 0:
                    metric_dataset_set.add((metric, dataset))
    with open(outpath, 'w') as fwrite:
        for metric in metric_set:
            fwrite.write('%s\tIsA\tmetric\n' % (metric))
        for metric, task in metric_task_set:
            fwrite.write('%s\tUsedFor\t%s\n' % (metric, task))
        for metric, dataset in metric_dataset_set:
            fwrite.write('%s\tUsedFor\t%s\n' % (metric, dataset))


def extract_datasets(inpath: str, outpath: str):
    dataset_set = set()
    dataset_task_set = set()
    with open(inpath) as fopen:
        data = json.load(fopen)
    for table in data:
        dataset = table['name'].lower()
        if len(dataset) > 0:
            dataset_set.add(dataset)
        for task_d in table['tasks']:
            task = task_d['task'].lower()
            if len(dataset) > 0 and len(task) > 0:
                dataset_task_set.add((dataset, task))
    with open(outpath, 'w') as fwrite:
        for dataset in dataset_set:
            fwrite.write('%s\tIsA\tdataset\n' % (dataset))
        for dataset, task in dataset_task_set:
            fwrite.write('%s\tUsedFor\t%s\n' % (dataset, task))


def extract_methods(inpath: str, outpath: str):
    collection_set = set()
    method_collection_set = set()
    method_area_set = set()
    with open(inpath) as fopen:
        data = json.load(fopen)
        for table in data:
            method = table['name'].lower()
            for col_d in table['collections']:
                collection = col_d['collection'].lower()
                area = col_d['area'].lower()
                if len(collection) > 0:
                    collection_set.add(collection)
                if len(method) > 0 and len(collection) > 0:
                    method_collection_set.add((method, collection))
                if len(method) > 0 and len(area) > 0:
                    method_area_set.add((method, area))
    with open(outpath, 'w') as fwrite:
        for col in collection_set:
            fwrite.write('%s\tIsA\tmethod\n' % (col))
        for method, col in method_collection_set:
            fwrite.write('%s\tIsA\t%s\n' % (method, col))
        for method, area in method_area_set:
            fwrite.write('%s\tUsedFor\t%s\n' % (method, area))


if __name__ == '__main__':
    data_dir = 'data/seed_taxonomy/paperswithcode'
    evaluation_table_path = '%s/evaluation-tables.json' % (data_dir)
    area_out_path = '%s/areas.txt' % (data_dir)
    # extract_areas(evaluation_table_path, area_out_path)

    tasks_out_path = '%s/tasks.txt' % (data_dir)
    # extract_tasks(evaluation_table_path, tasks_out_path)

    metrics_out_path = '%s/metrics.txt' % (data_dir)
    # extract_metrics(evaluation_table_path, metrics_out_path)

    dataset_in_path = '%s/datasets.json' % (data_dir)
    dataset_out_path = '%s/datasets.txt' % (data_dir)
    # extract_datasets(dataset_in_path, dataset_out_path)

    method_in_path = '%s/methods.json' % (data_dir)
    method_out_path = '%s/methods.txt' % (data_dir)
    extract_methods(method_in_path, method_out_path)
