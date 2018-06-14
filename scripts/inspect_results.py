
# coding: utf-8

# In[2]:


import pymongo
from IPython.display import display

import glob
import os
import json


def find_runs_on_filesystem(campaign_name, logs_filepath="../experiment-logs/"):

    runs = []
    for run_dir in glob.glob("/".join([logs_filepath, "[0-9]*"])):
        run = {}
        try:
            with open(os.path.join(run_dir, "info.json"), "r") as f:
                run["info"] = json.load(f)
            with open(os.path.join(run_dir, "config.json"), "r") as f:
                run["config"] = json.load(f)
            if run["config"]["experiment_name"] == campaign_name:
                runs.append(run)
        except IOError as e:
            print(e)
    return runs

def report_results_of_a_specific_campaign(campaign_name, db_type):

    print(campaign_name)
    if db_type == "mongo":
        client = pymongo.MongoClient("localhost", 27017)
        db = client.joint_ner_and_md
        runs = db.runs.find({"config.experiment_name": campaign_name})
    else:
        runs = find_runs_on_filesystem(campaign_name, logs_filepath=db_type)

    configs = []
    for run_idx, run in enumerate(runs):

        dict_to_report = dict(run["config"])
        initial_keys = dict_to_report.keys()

        print initial_keys

        result_designation_labels = ["MORPH", "NER", "YURET"]

        dict_to_report["epochs"] = max([len(run["info"][label].keys())
                                        for label in ["NER_dev_f_score", "MORPH_dev_f_score"]])

        for result_designation_label in result_designation_labels:

            print "result_designation_label: ", result_designation_label

            if result_designation_label == "YURET":
                best_performances = run["info"][result_designation_label + "_test_f_score"]
            else:
                best_performances = run["info"][result_designation_label + "_dev_f_score"]
            print best_performances
            best_dev_result_for_this_run = 0
            best_test_result_for_this_run = 0
            epoch_id_of_the_best_dev_result = -1
            # display(run["config"])
            for epoch in sorted([int(k) for k in best_performances.keys()]):
                # if result_designation_label != "NER":
                #     corrected_epoch = epoch + 1
                epoch_max = max(best_performances[str(epoch)])
                if epoch_max > best_dev_result_for_this_run:
                    epoch_id_of_the_best_dev_result = epoch
                    best_dev_result_for_this_run = epoch_max
                    best_test_result_for_this_run = \
                        max(run["info"][result_designation_label + "_test_f_score"][str(epoch)])

                # print "run_idx: %d, epoch: %d, epoch_best_performance: %.2lf, best_for_this_run: %.2lf" % (run_idx, epoch, epoch_max, best_for_this_run)

            dict_to_report[result_designation_label + "_best_dev"] = best_dev_result_for_this_run
            dict_to_report[result_designation_label + "_best_test"] = best_test_result_for_this_run

            for x in result_designation_labels:
                # if x != result_designation_label:
                print "x: ", x
                print "epoch_id_of_the_best_dev_result: ", epoch_id_of_the_best_dev_result
                dict_to_report[result_designation_label + "_to_" + x + "_test"] = \
                    max(run["info"][x + "_test_f_score"][str(epoch_id_of_the_best_dev_result)]) \
                        if str(epoch_id_of_the_best_dev_result) in run["info"][x + "_test_f_score"].keys() else -1
                print dict_to_report[result_designation_label + "_to_" + x + "_test"]

        configs.append({key: dict_to_report[key] for key in [x for x in ["host",
                                                             "integration_mode",
                                                             "active_models",
                                                             "train_with_yuret",
                                                             "use_golden_morpho_analysis_in_word_representation",
                                                             "multilayer",
                                                             "shortcut_connections",
                                                             "epochs"] if x in dict_to_report] +
                        [x for x in dict_to_report.keys() if x not in initial_keys]})

    import pandas
    df = pandas.DataFrame.from_dict(configs)
    print configs
    cols = df.columns.tolist()

    # display(df[["host"] +
    #                     [x for x in dict_to_report.keys() if x not in initial_keys]])

    display(df)

    df_groupedby_hyperparameters = df.groupby(["integration_mode",
                                             "active_models",
                                             "train_with_yuret",
                                             "use_golden_morpho_analysis_in_word_representation",
                                             "multilayer",
                                             "shortcut_connections"])
    return df, df_groupedby_hyperparameters.NER_best_test.mean()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--campaign_name", default="section1-all-20171013-01")

    parser.add_argument("--db_type", default="mongo")

    args = parser.parse_args()

    df, df_groupedby_hyperparameter_NER_best_test_mean = report_results_of_a_specific_campaign(args.campaign_name, args.db_type)
    df.to_csv("./scripts/results-%s.csv" % args.campaign_name)
    df_groupedby_hyperparameter_NER_best_test_mean.to_csv("./scripts/results-NER_best_test_mean-%s.csv" % args.campaign_name)