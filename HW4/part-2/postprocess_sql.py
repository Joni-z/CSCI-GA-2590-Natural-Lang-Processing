import argparse
import pickle
import re

from load_data import load_lines
from utils import compute_record


TIME_COLUMNS = (
    "arrival_time",
    "departure_time",
    "time_elapsed",
    "stops",
    "connections",
    "round_trip_cost",
    "one_direction_cost",
    "ground_fare",
    "capacity",
)


def infer_operator(nl_query):
    query = nl_query.lower()
    if any(phrase in query for phrase in ("before", "earlier than", "less than", "prior to")):
        return "<"
    if any(phrase in query for phrase in ("after", "later than", "greater than")):
        return ">"
    return "="


def remove_duplicate_from_aliases(sql_query):
    match = re.search(r"\bFROM\b(.*?)\bWHERE\b", sql_query, flags=re.IGNORECASE)
    if match is None:
        return sql_query

    seen_aliases = set()
    kept_items = []
    for item in [item.strip() for item in match.group(1).split(",")]:
        alias_match = re.search(r"\b(\w+_\d+)\b$", item)
        alias = alias_match.group(1) if alias_match else item
        if alias not in seen_aliases:
            seen_aliases.add(alias)
            kept_items.append(item)

    return (
        sql_query[: match.start(1)]
        + " "
        + ", ".join(kept_items)
        + " "
        + sql_query[match.end(1) :]
    )


def add_missing_comparison_operators(sql_query, nl_query):
    columns = "|".join(TIME_COLUMNS)
    pattern = rf"(\b\w+_\d+\.(?:{columns})\b)\s{{2,}}(\d+)"
    operator = infer_operator(nl_query)
    return re.sub(pattern, lambda match: f"{match.group(1)} {operator} {match.group(2)}", sql_query)


def balance_small_syntax_errors(sql_query):
    sql_query = sql_query.strip()

    if sql_query.endswith(")") and sql_query.count(")") > sql_query.count("("):
        sql_query = sql_query[:-1].rstrip()

    missing_parens = sql_query.count("(") - sql_query.count(")")
    if 0 < missing_parens <= 4:
        sql_query += ")" * missing_parens

    if sql_query.count("'") % 2:
        sql_query += "'"

    return re.sub(r"\s+", " ", sql_query).strip()


def postprocess_query(sql_query, nl_query, had_error):
    if not had_error:
        return sql_query

    sql_query = remove_duplicate_from_aliases(sql_query)
    sql_query = add_missing_comparison_operators(sql_query, nl_query)
    return balance_small_syntax_errors(sql_query)


def postprocess_file(args):
    nl_queries = load_lines(args.nl)
    sql_queries = load_lines(args.input_sql)
    with open(args.input_records, "rb") as f:
        records, errors = pickle.load(f)

    processed_queries = []
    processed_records = list(records)
    processed_errors = list(errors)
    changed_indices = []

    for i, (nl_query, sql_query, error) in enumerate(zip(nl_queries, sql_queries, errors)):
        processed_query = postprocess_query(sql_query, nl_query, bool(error))
        processed_queries.append(processed_query)
        if processed_query != sql_query:
            _, record, error_msg = compute_record(i, processed_query)
            processed_records[i] = record
            processed_errors[i] = error_msg
            changed_indices.append(i)

    with open(args.output_sql, "w") as f:
        for query in processed_queries:
            f.write(f"{query}\n")

    with open(args.output_records, "wb") as f:
        pickle.dump((processed_records, processed_errors), f)

    print(f"Changed {len(changed_indices)} queries")
    print(f"Remaining SQL errors: {sum(bool(error) for error in processed_errors)}")


def get_args():
    parser = argparse.ArgumentParser(description="Conservative SQL cleanup for generated T5 queries.")
    parser.add_argument("--nl", required=True)
    parser.add_argument("--input_sql", required=True)
    parser.add_argument("--input_records", required=True)
    parser.add_argument("--output_sql", required=True)
    parser.add_argument("--output_records", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    postprocess_file(get_args())
