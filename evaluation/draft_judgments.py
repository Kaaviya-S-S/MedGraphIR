"""Create reviewable, rule-based draft grades for the pooled candidates.

These suggestions are not human judgments. Review each row, correct the
relevance grade when needed, and set human_reviewed=yes only after review.
"""

import pandas as pd

POOL_FILE = "evaluation/judgment_candidates.csv"

# (entities that must appear in the record, expected relation or None)
QUERY_INTENTS = {
    "Find medicines that interact with Warfarin.": (("Warfarin",), "INTERACTS_WITH"),
    "What are the documented interaction partners of Warfarin?": (("Warfarin",), "INTERACTS_WITH"),
    "Which medications have reported interactions with Tacrolimus?": (("Tacrolimus",), "INTERACTS_WITH"),
    "List drugs known to interact with Clarithromycin.": (("Clarithromycin",), "INTERACTS_WITH"),
    "What interactions are recorded for Dexamethasone?": (("Dexamethasone",), "INTERACTS_WITH"),
    "Find reported interaction partners for Doripenem.": (("Doripenem",), "INTERACTS_WITH"),
    "Are there any records about interactions with Ferumoxides?": (("Ferumoxides",), "INTERACTS_WITH"),
    "Which medicines are listed as interacting with Bismuth subcitrate potassium?": (
        ("Bismuth subcitrate potassium",), "INTERACTS_WITH",
    ),
    "What diseases are listed for Metformin?": (("Metformin",), "TREATS"),
    "What is Metformin used for according to the records?": (("Metformin",), "TREATS"),
    "Find diseases linked to Methotrexate.": (("Methotrexate",), "TREATS"),
    "Which conditions does Prednisone treat?": (("Prednisone",), "TREATS"),
    "Are there indications for Omeprazole?": (("Omeprazole",), "TREATS"),
    "Tell me about the recorded relationships for Cyclosporine.": (("Cyclosporine",), None),
    "Which drugs are listed for Hypertension?": (("Hypertension",), "TREATS"),
    "What medicines treat Asthma?": (("Asthma",), "TREATS"),
    "Show treatments listed for Kawasaki disease.": (("Kawasaki disease",), "TREATS"),
    "Which drugs are associated with Gestational diabetes?": (("Gestational diabetes",), "TREATS"),
    "Find treatments for Barrett's esophagus.": (("Barrett's esophagus",), "TREATS"),
    "Which drugs are linked to Nephrolithiasis?": (("Nephrolithiasis",), "TREATS"),
    "What relationships are recorded for Metformin?": (("Metformin",), None),
    "Show all connections for Cyclosporine.": (("Cyclosporine",), None),
    "Is there a recorded relationship between Warfarin and Acetylsalicylic acid?": (
        ("Warfarin", "Acetylsalicylic acid"), "INTERACTS_WITH",
    ),
    "Are any diseases associated with Acetylsalicylic acid?": (("Acetylsalicylic acid",), "TREATS"),
    "What conditions are related to Omeprazole?": (("Omeprazole",), "TREATS"),
    "Which drug relationships mention Kawasaki disease?": (("Kawasaki disease",), "TREATS"),
    "Does the graph include any information on Agalsidase beta?": (("Agalsidase beta",), None),
    "What is documented about Methotrexate and its relationships?": (("Methotrexate",), None),
    "Which medicines have interactions with Worfarin?": (("Warfarin",), "INTERACTS_WITH"),
    "What therapeutic indications are recorded for Metfromin?": (("Metformin",), "TREATS"),
}


def _grade(row: pd.Series) -> tuple[int, str]:
    required_entities, expected_relation = QUERY_INTENTS[row["query"]]
    endpoints = {
        str(row["source_entity"]).casefold(),
        str(row["target_entity"]).casefold(),
    }
    matched = sum(entity.casefold() in endpoints for entity in required_entities)

    if expected_relation is None:
        if matched == len(required_entities):
            return 2, "Contains all query entities; the query asks for any recorded relation."
        if matched:
            return 1, "Contains one named entity from the query, but not all requested entities."
        return 0, "Does not contain the query's named entity."

    if matched == len(required_entities) and row["relationship"] == expected_relation:
        return 2, "Named entity/entities and requested relationship type match."
    if matched:
        return 1, "Contains a named entity, but the requested relation or entity pair does not fully match."
    return 0, "Does not contain the query's named entity."


def main() -> None:
    pool = pd.read_csv(
        POOL_FILE,
        keep_default_na=False,
        dtype={"query": str, "record_id": str},
    )
    required_columns = {"query", "record_id", "source_entity", "relationship", "target_entity"}
    missing = required_columns - set(pool.columns)
    if missing:
        raise ValueError(f"{POOL_FILE} is missing columns: {sorted(missing)}")
    unknown_queries = set(pool["query"]) - QUERY_INTENTS.keys()
    if unknown_queries:
        raise ValueError(f"No draft-judgment rule for queries: {sorted(unknown_queries)}")

    if "relevance" not in pool:
        pool["relevance"] = ""
    if "human_reviewed" not in pool:
        pool["human_reviewed"] = ""

    drafts = pool.apply(_grade, axis=1)
    draft_grades = [grade for grade, _ in drafts]
    draft_reasons = [reason for _, reason in drafts]
    if "draft_reason" not in pool:
        pool["draft_reason"] = ""

    for position in range(len(pool)):
        existing_grade = str(pool.at[position, "relevance"]).strip()
        if not existing_grade:
            pool.at[position, "relevance"] = draft_grades[position]
            pool.at[position, "draft_reason"] = draft_reasons[position]
        elif (
            not str(pool.at[position, "draft_reason"]).strip()
            and existing_grade == str(draft_grades[position])
        ):
            pool.at[position, "draft_reason"] = draft_reasons[position]

    if pool["human_reviewed"].astype(str).str.strip().ne("").any():
        print("[DRAFT] Preserved existing human_reviewed values.")
    pool.to_csv(POOL_FILE, index=False)

    print(f"[DRAFT] Wrote {len(pool):,} suggested grades to {POOL_FILE}")
    print("[DRAFT] These are not human judgments. Review and correct every grade.")
    print(pool["relevance"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
