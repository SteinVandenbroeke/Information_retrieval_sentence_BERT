import datetime
import os
import pickle

from tqdm import tqdm
import pandas as pd

from src.queryProcessor import QueryProcessor


def test_queries(queryProcessing, small = True):
    """
    test a dataset with given queries and results
    :param queryProcessing: the query processor to use
    :param small: whether to use the small or big dataset
    :return: the passed queries and the amount of queries tested
    """
    print("Testing queries for ", "small" if small else "big", " database:")
    query_path = "../queries/dev_queries" + ("_small.csv" if small else ".csv")
    query_data = pd.read_csv(query_path, names=["Query number","Query"], skiprows=1)
    query_result_path = "../queries/dev_query_results" + ("_small.csv" if small else ".csv")
    query_result_data = pd.read_csv(query_result_path, names=["Query_number","doc_number"], skiprows=1)

    passes = 0
    fails = 0

    for q in query_data.values:
        query_id = q[0]
        query = q[1]
        result = []
        for i in queryProcessing.processQuery(query,1):
            result.append(int(i.replace(".txt","").split("output_")[1]))

        query_result_doc = query_result_data[query_result_data['Query_number'] == query_id]['doc_number'].tolist()
        intersect = list(set(result).intersection(set(query_result_doc)))
        print("matched items", intersect)

        if len(intersect) > 0:
            passes += 1
            print("passed")
        else:
            fails += 1
            print("failed")
    print(passes, " out of ", passes+fails, " queries passed.")
    return (passes, passes+fails)

def create_result_csv(queryProcessing, small = True):
    """
    create a result csv file containing the queries and their results
    :param queryProcessing: the query processor to use
    :param small: wheter to use the small or big dataset
    :return:
    """
    start_time = datetime.datetime.now()

    print("Testing queries for ", "small" if small else "big", " database:")
    query_path = "../queries/queries_test_set.csv"
    query_data = pd.read_csv(query_path, names=["Query number","Query"], skiprows=1, delimiter='\t')

    passes = 0
    fails = 0
    if os.path.isfile("../results/result.csv"):
        os.remove("../results/result.csv")
    with open('../results/result.csv', 'w') as file:
        file.write(f"Query_number,doc_number\n")
        for q in tqdm(query_data.values):
            query_id = q[0]
            query = q[1]
            results = []
            for i in queryProcessing.processQuery(query,10):
                results.append(int(i.replace(".txt", "").split("output_")[1]))

            for result in results:
                file.write(f"{int(query_id)},{int(result)}\n")
    end_time_query_test = datetime.datetime.now()
    print("test csv time: ", end_time_query_test - start_time)

def evaluation(query_processing: QueryProcessor, small=True, sub_folder="", add_to_file_name = ""):
    """
    calculates the MAP and MAR for different k values and prints the results
    :param query_processing:  the query processor to use
    :param small: whether to use the small or big dataset
    :return: the resulted MAP and MAR
    """
    start_time = datetime.datetime.now()
    add_text = ""
    if hasattr(query_processing, "t") and query_processing.t != None:
        add_text = "_c_" + str(query_processing.t)
    if not os.path.isdir("../evaluations/" + sub_folder + "/"):
        os.makedirs("../evaluations/"  + sub_folder + "/")
    evalutation_file = "../evaluations/"  + sub_folder  + "/" + "eval_" + query_processing.file_name + "_" + str(small)  + add_text + add_to_file_name + ".txt"
    if os.path.isfile(evalutation_file):
        evaluation_result = pickle.load(open(evalutation_file, "rb"))
        print("Query time: ", evaluation_result["QueryTime"])
        print(f"MAP@1: {evaluation_result["MAP@1"]} MAR@1: {evaluation_result["MAR@1"]}")
        print(f"MAP@3: {evaluation_result["MAP@3"]} MAR@3: {evaluation_result["MAR@3"]}")
        print(f"MAP@5: {evaluation_result["MAP@5"]} MAR@5: {evaluation_result["MAR@5"]}")
        print(f"MAP@10: {evaluation_result["MAP@10"]} MAR@10: {evaluation_result["MAR@10"]}")
        return evaluation_result

    print("Testing queries for ", "small" if small else "big", " database:")
    query_path = "../queries/dev_queries" + ("_small.csv" if small else ".csv")
    query_data = pd.read_csv(query_path, names=["Query number", "Query"], skiprows=1)
    query_result_path = "../queries/dev_query_results" + ("_small.csv" if small else ".csv")
    query_result_data = pd.read_csv(query_result_path, names=["Query_number", "doc_number"], skiprows=1)

    sumedP10 = 0.0
    sumedR10 = 0.0
    sumedP5 = 0.0
    sumedR5 = 0.0
    sumedP3 = 0.0
    sumedR3 = 0.0
    sumedP1 = 0.0
    sumedR1 = 0.0
    total = 0
    for q in tqdm(query_data.values[:1000]):
        query_id = q[0]
        query = q[1]
        results = []
        for i in query_processing.processQuery(query,10):
            results.append(int(i.replace(".txt","").split("output_")[1]))
        query_result_doc = query_result_data[query_result_data['Query_number'] == query_id]['doc_number'].tolist()
        #print("Result doc", 10, query_id, query_result_doc, results)
        r = len(list(set(results).intersection(set(query_result_doc))))
            #print(results, query_result_doc, r, f"MAP@10: {r/1} MAR@10: {r/len(query_result_doc)}")
        sumedP10 += r/10
        sumedR10 += r/len(query_result_doc)

        results = results[0:5]
        #print("Result doc", 5, query_id, query_result_doc, results)
        r = len(list(set(results).intersection(set(query_result_doc))))
        sumedP5 += r / 5
        sumedR5 += r / len(query_result_doc)

        results = results[0:3]
        #print("Result doc", 3, query_id, query_result_doc, results)
        r = len(list(set(results).intersection(set(query_result_doc))))
        sumedP3 += r / 3
        sumedR3 += r / len(query_result_doc)

        result = results[0]
        #print("Result doc", 3, query_id, query_result_doc, results)
        r = result in query_result_doc
        sumedP1 += r / 1
        sumedR1 += r/len(query_result_doc)

        total += 1
        # if total % 100 == 0:
        #     print("done evaluating: ", total)

    end_time_query_test = datetime.datetime.now()

    evaluation_result = {
        "QueryTime": end_time_query_test - start_time,
        "MAP@1": sumedP1 / total,
        "MAR@1": sumedR1 / total,
        "MAP@3": sumedP3 / total,
        "MAR@3": sumedR3 / total,
        "MAP@5": sumedP5 / total,
        "MAR@5": sumedR5 / total,
        "MAP@10": sumedP10 / total,
        "MAR@10": sumedR10 / total,
    }

    print("Query time: ", evaluation_result["QueryTime"])
    print(f"MAP@1: {evaluation_result["MAP@1"]} MAR@1: {evaluation_result["MAR@1"]}")
    print(f"MAP@3: {evaluation_result["MAP@3"]} MAR@3: {evaluation_result["MAR@3"]}")
    print(f"MAP@5: {evaluation_result["MAP@5"]} MAR@5: {evaluation_result["MAR@5"]}")
    print(f"MAP@10: {evaluation_result["MAP@10"]} MAR@10: {evaluation_result["MAR@10"]}")

    pickle.dump(evaluation_result, open(evalutation_file, "wb"))
    return evaluation_result

