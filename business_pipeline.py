import json
import os
import time
import re
from tqdm import tqdm
import pandas as pd
import spacy
from openai import OpenAI

# Load OpenAI API credentials
with open("config/openai_credentials.txt") as f:
    sk = json.load(f)

# Initialize OpenAI client using the API key
client = OpenAI(api_key=sk["api_key"])

allowed_predicates = {
    "acquired_by": "e2 purchases controlling stake in e1. The relation is directed. The inverted relation is best described by the same relation type.",
    "brand_of": "e2 offers products or services of e1 (Brand). The relation is directed. The inverted relation is best described by the same relation type.",
    "client_of": "e1 uses (and presumably pays for) products or services offered by e2. The relation is directed. The inverted relation is best described by â€œSupplier ofâ€.",
    "collaboration": "e1 and e2 collaborate in (parts of their) business activities. The relation is undirected.",
    "merge_with": "e1 and e2 merged (parts of) their business activities. The relation is undirected.",
    "product_or_service_of": "e1 is offered for commercial distribution by e2. The relation is directed. The inverted relation is best described by the same relation type.",
    "shareholder_of": "e1 owns shares in e2. The relation is directed. The inverted relation is best described by the same relation type.",
    "subsidiary_of": "e2 legally owns e1. The relation is directed. The inverted relation is best described by â€œParent ofâ€.",
    "traded_on": "Shares of e1 are listed on e2 (Stock exchange). The relation is directed. The inverted relation is best described by â€œlistsâ€."
}

example_table_1 = {
        "Year": [
          "1995",
          "1997",
          "1998"
        ],
        "Title": [
          "Friday",
          "Dangerous Ground",
          "The Players Club"
        ],
        "Director": [
          "F. Gary Gray",
          "Darrell Roodt",
          "Ice Cube"
        ],
        "Distributor": [
          "New Line Cinema",
          "New Line Cinema",
          "New Line Cinema"
        ],
        "Notes": []
      }

example_text_1 = (
     "Ice Cubestarted his movie producing career in 1995 with his then manager Patricia Charbonet. Together, they producedFriday(1995),Dangerous Ground(1997) andThe Players Club(1998). Cube, along with new producing partner Matt Alvarez, founded CubeVision (later credited in films as Cube Vision) in 1998.[1]The company's first film would be 2000'sNext Friday, a sequel to Ice Cube's 1995 filmFriday. Cube Vision went on to produceAll About the Benjamins,BarbershopandFriday After Next, the third film in theFridayfilm series, in 2002."
)

example_mapping_1 = {
    "Distributor": "client_of",
    "Title": "product_or_service_of",
    "Director": None,
    "Year": None,
    "Notes": None
}

example_table_2 = {
        "Title": [
          "Hello Stranger: The Movie",
          "The Death of Nintendo",
          "Whether the Weather is Fine"
        ],
        "Release date": [
          "February 12",
          "April 23 (online)",
          "December 25"
        ],
        "Director": [
          "Dwein Baltazar",
          "Raya Martin",
          "Carlo Francisco Manatad"
        ],
        "Cast": [
          "Tony Labrusca, JC Alcantara, Vivoree Esclito, Patrick Quiroz, Gillian Vicencio, Miguel Almendras",
          "Noel Comia Jr.",
          "Charo Santos-Concio, Daniel Padilla, Rans Rifol"
        ],
        "Genre(s)": [
          "Boys' Love",
          "Horror",
          "Drama"
        ],
        "Associated film production": [
          "IndieFlip Toho (Japan)",
          "Globe Studios, Dreamscape Entertainment"
        ]
      }

example_text_2 = (
   "Black Sheep is best known for producing the filmsExes Baggage(2018),Alone/Together(2019),Fan Girl(2020), andWhether the Weather is Fine(2021)..\n\nIndieFlipToho(Japan)"
)

example_mapping_2 = {
    "Title": "product_or_service_of",
    "Release date": None,
    "Director": None,
    "Cast": None,
    "Genre(s)": None,
    "Associated film production": "collaboration"
}

example_table_3 = {
        "Industry": [
          "Aerospace"
        ],
        "Predecessor": [
          "Stoddard-Hamilton Aircraft"
        ],
        "Founded": [
          "2001"
        ],
        "Founders": [
          "Tom Hamilton and Thomas Wathen"
        ],
        "Headquarters": [
          "Arlington, Washington"
        ],
        "Key people": [
          "CEO: Randy Lervold"
        ],
        "Products": [
          "Homebuilt aircraft kits"
        ],
        "Owner": [
          "Jilin Hanxing Group"
        ],
        "Website": [
          "glasairaviation.com"
        ]
      }

example_text_3 = (
    "Tom Hamilton began flight testing theGlasair TDand foundedStoddard-Hamilton Aircraftin 1979. Glasair Aviation was formed in 2001 when Thomas W. Wathen purchased the Glasair assets from bankrupt Stoddard-Hamilton Aircraft, Inc. and signed an agreement with Arlington Aircraft Development, Inc. (AADI) to buy all rights to and assets of the GlaStar model.[1][2].\n\nIn July 2012 the company was sold to the Jilin Hanxing Group, which formed a new company Glasair Aircraft USA, LLC. The company indicated that it intended tocertifythe Glastar design and otherwise retain production in Arlington, Washington. Its chairman said that purchasing Glasair was \"the first step in a very long journey\" and envisioned the company producingtrainersfor flight schools and eventually personal aircraft for the Chinese market.[3][4]"
)

example_mapping_3 = {
    "Industry": None,
    "Predecessor": None,
    "Founded": None,
    "Founders": None,
    "Headquarters": None,
    "Key people": None,
    "Products": None,
    "Owner": "acquired_by",
    "Website": None
}

example_text_4 = (
    "LocusPoint Networks LLCwas an owner of television stations in theUnited States. The company is 99% owned byThe Blackstone Group. After selling off most of their in 2017 and 2018 in theFederal Communications Commission(FCC) spectrum auction and to other broadcasters, they owned one remaining station, WLEP-LD inErie, Pennsylvania, whose license they turned in to the FCC effective February 12, 2019."
)

example_table_4 = {
        "Company type": [
          "Private"
        ],
        "Industry": [
          "Broadcast television"
        ],
        "Founded": [
          "2012"
        ],
        "Defunct": [
          "2019"
        ],
        "Headquarters": [
          "Pleasanton, California, United States"
        ],
        "Key people": [
          "Ravi Patharlanka, President"
        ],
        "Owner": [
          "Blackstone Group (99%)"
        ],
        "Website": [
          "locuspointnetworks.com"
        ]
      }

example_mapping_4 = {
    "Company type": None,
    "Industry": None,
    "Founded": None,
    "Defunct": None,
    "Headquarters": None,
    "Key people": None,
    "Owner": "subsidiary_of",
    "Website": None
  }

example_mapping_5 = {
  "Co-production with": "collaboration",
  "Network": "client_of",
  "Title": "product_or_service_of",
  "Released": None
}

example_text_5 = (
  "Silvergate Media was created in 2011 as part of amanagement buyout, when Alli purchased the rights toOctonautsandThe World of Beatrix PotterfromChorion, a company he was previously chair of.[3]"
)

example_table_5 = {
        "Title": [
          "Hilda and the Mountain King"
        ],
        "Released": [
          "2021"
        ],
        "Network": [
          "Netflix"
        ],
        "Co-production with": [
          "Mercury Filmworks and Nobrow Press"
        ]
      }

example_text_6 = (
   "SES Americomwas a major commercialsatelliteoperator of North Americangeosynchronous satellitesbased in theUnited States. The company started asRCA Americomin 1975 before being bought byGeneral Electricin 1986 and then later acquired bySESin 2001. In September 2009, SES Americom andSES New Skiesmerged intoSES World Skies.[2].\n\nSatcom 1 was instrumental in helping early cable TV channels (such asSuperstation TBSandCBN) to become initially successful, because these channels distributed their programming to all of the local cable TVheadendsusing the satellite. Additionally, it was the first satellite used by broadcast TV networks in the United States, likeAmerican Broadcasting Company(ABC),NBC, andCBS, to distribute their programming to all of their local affiliate stations.Satcom 1was so widely used because it had twice the communications capacity of the competingWestar 1(24 transponders as opposed to Westar 1's 12), which resulted in lower transponder usage costs. 14 more (increasingly sophisticated) Satcom satellites would enter service from 1976 to 1992..\n\nIn November 2001, GE sold its GE Americom unit toSESfor US$5 billion in cash and stock. As a result of the sale, GE Americom was renamed SES Americom and SES Global was formed as the parent company. SES's existing operations were moved to the newly created SES Astra subsidiary.[3][4]SES formerly bought a satellite from failedDirect broadcast satellite(DBS) company Crimson Satellite Associates and GE Americom while still under construction byGE AstroSpace(asSatcom K3).[5]RenamedAstra 1Band modified for use as a European direct broadcasting satellite and a part of the Astra DBS constellation, it was launched to add extra capacity to thesatellite televisionservices from19.2° East, servingGermany, theUnited KingdomandRepublic of Ireland..\n\nAfter the acquisition of GE Americom by SES, all the satellites previously named with the GE-# prefix were renamed AMC-# (i.e., GE-1 renamed AMC-1, and so on).[6]"
)

example_table_6 = {
        "Satellite": [
          "AMC-1",
          "AMC-2",
          "AMC-3",
          "AMC-4",
          "AMC-5",
          "AMC-6",
          "AMC-7",
          "AMC-8",
          "AMC-9",
          "AMC-10",
          "AMC-11",
          "AMC-12",
          "AMC-14[18]",
          "AMC-15",
          "AMC-16",
          "AMC-18",
          "Satcom C3",
          "AMC-21"
        ],
        "Position": [
          "131° West",
          "101° West",
          "87° West",
          "101° West",
          "79° West",
          "72° West",
          "137° West",
          "139° West",
          "83° West",
          "135° West",
          "131° West",
          "37° West",
          "61.5° West (planned)",
          "105° West",
          "85° West",
          "139° West",
          "79° West",
          "125° West"
        ],
        "Manufacturer": [
          "Lockheed Martin",
          "Lockheed Martin",
          "Lockheed Martin",
          "Lockheed Martin",
          "Alcatel Space",
          "Lockheed Martin",
          "Lockheed Martin",
          "Lockheed Martin",
          "Alcatel Alenia Space",
          "Lockheed Martin",
          "Lockheed Martin",
          "Alcatel Alenia Space",
          "Lockheed Martin",
          "Lockheed Martin",
          "Lockheed Martin",
          "Lockheed Martin",
          "GE AstroSpace",
          "Thales Alenia Space / Orbital Sciences Corporation"
        ],
        "Model": [
          "A2100A",
          "A2100A",
          "A2100A",
          "A2100AX",
          "Spacebus 2000",
          "A2100AX",
          "A2100A",
          "A2100A",
          "Spacebus 3000B3",
          "A2100A",
          "A2100A",
          "Spacebus 4000C3",
          "A2100",
          "A2100AX",
          "A2100AX",
          "A2100A",
          "GE-3000",
          "STAR-2"
        ],
        "Launched": [
          "8 September 1996",
          "30 January 1997",
          "4 September 1997",
          "13 November 1999",
          "28 October 1998",
          "22 October 2000",
          "14 September 2000",
          "19 December 2000",
          "7 June 2003",
          "5 February 2004",
          "19 May 2004",
          "3 February 2005",
          "14 March 2008",
          "15 October 2004",
          "17 December 2004",
          "8 December 2006",
          "10 September 1992",
          "14 August 2008"
        ],
        "Launch vehicle": [
          "Atlas IIA",
          "Ariane 44L",
          "Atlas IIAS",
          "Ariane 44LP",
          "Ariane 44L",
          "Proton-K / DM-2",
          "Ariane 5G",
          "Ariane 5G",
          "Proton-K / Briz-M[12]",
          "Atlas IIAS[14]",
          "Atlas IIAS[15]",
          "Proton-M / Briz-M[16]",
          "Proton-M / Briz-M",
          "Proton-M / Briz-M[20]",
          "Atlas V (521) [21]",
          "Ariane 5 ECA",
          "Ariane 44LP",
          "Ariane 5 ECA[22]"
        ],
        "Comments": [
          "[citation needed]",
          "Replaced by SES-1[11]",
          "[citation needed]",
          "Launched in 1999 as GE-4. Replaced by SES-1[11]",
          "[citation needed]",
          "[citation needed]",
          "Launched in 2000 as GE-7. Backup to AMC-10 since 2015",
          "Launched in 2000 as GE-8",
          "Failed in June 2017, apparently broke apart [13]",
          "Renamed NSS-10 [17]",
          "Launch failure [19]",
          "Replaced AMC-2 previously at 105° West",
          "Graveyard orbit"
        ]
      }

example_mapping_6 = {
  "Satellite": "acquired_by",
  "Manufacturer": "None",
  "Position": "None",
  "Model": "None",
  "Launched": "None",
  "Launch vehicle": "None",
  "Comments": "None"
}

# =========================
# SPACY MODEL
# =========================
nlp = spacy.load("en_core_web_trf")

reverse_columns = {"parent", "owner"}
numeric_pattern = re.compile(r"^\d+(\.\d+)?$")

def extract_entities(text):
    if not text or str(text).strip() == "":
        return []

    text = str(text).strip()
    text = re.sub(r"\[\d+\]|\(\d{4}(-\d{4})?\)", "", text).strip()

    doc = nlp(text)
    entities = [ent.text.strip() for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "GPE"]]

    if not entities and not numeric_pattern.match(text):
        entities = [text]

    return [e for e in entities if e]


def clean_title(title):
    if not title:
        return ""
    title = str(title).strip()
    title = re.sub(r"\s*\d+$", "", title)
    return title.strip()

def safe_json_extract(raw_text):

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)

    if match:
        raw_text = match.group(1)

    try:
        obj = json.loads(raw_text)
    except:
        obj = {}

    for key in ["Predicate mapping", "Predicate_mapping"]:
        if key in obj:
            obj = obj[key]

    obj.pop("Explanation", None)

    return obj

# ============================================================
# STAGE 1: COLUMN -> PREDICATE MAPPING
# ============================================================

def get_mapping(new_table, new_text, new_title):
    prompt_text = f"""
    You are an AI assistant that maps table columns to knowledge graph predicates.

    Allowed predicates:
    {chr(10).join([f"- {k}: {v}" for k, v in allowed_predicates.items()])}

    Additional Guidance:
    1. Acquisition vs Subsidiary:
    - Past tense phrases in text (e.g., "was sold to", "acquired by") → acquired_by
    - Present tense phrases (e.g., "is a subsidiary of", "part of", "owned by") → subsidiary_of
    - If text is missing or ambiguous, but the table lists an owner → subsidiary_of.

    2. Collaboration:
    - Look for joint activity keywords in table or text: "co-produced", "collaboration", "joint venture", "partnered with" → collaboration

    3. Attribute-only columns:
    - Columns like "Founded", "Industry", "Headquarters", "Key people", "Website", "Products" → None (no relational predicate)

    --------------------------------------------------
    Example 1 (Cube Vision)
    Table columns:
    {chr(10).join([f"- {k}: {v}" for k, v in example_table_1.items()])}
    Text:
    {example_text_1}
    Relevant entity: CubeVision (may appear as subject or object)
    Explanation:
    - "Title":
        - Table: Lists film titles associated with CubeVision
        - Text: Confirms CubeVision produced these films
        - Reasoning → Films are products/services of CubeVision → product_or_service_of
    - "Distributor":
        - Table: Distributor column lists New Line Cinema
        - Text: Mentions CubeVision’s films were produced and presumably distributed by these partners
        - Reasoning → Distributor provides services used by CubeVision → client_of
    - "Director", "Year", "Notes":
        - Table: Attribute information, not external relationships
        - Text: Does not indicate relational information
        - Reasoning → No relational predicate → None
    Predicate mapping:
    {json.dumps(example_mapping_1, indent=2)}

    --------------------------------------------------
    Example 2 (Black Sheep Productions)
    Table columns:
    {chr(10).join([f"- {k}: {v}" for k, v in example_table_2.items()])}
    Text:
    {example_text_2}
    Relevant entity: Black Sheep (may appear as subject or object)
    Explanation:
    - "Title":
        - Table: Lists movie titles produced by Black Sheep
        - Text: Mentions Black Sheep is producing these films
        - Reasoning → These are products/services of Black Sheep → product_or_service_of
    - "Associated film production":
        - Table: Lists collaborating production companies
        - Text: Confirms collaboration with IndieFlip Toho
        - Reasoning → Joint production indicates collaboration → collaboration
    - "Release date", "Director", "Cast", "Genre(s)":
        - Table: Attribute information, not external relationships
        - Text: No relational information
        - Reasoning → No relational predicate → None
    Predicate mapping:
    {json.dumps(example_mapping_2, indent=2)}

    --------------------------------------------------
    Example 3 (Glasair Aviation)
    Table columns:
    {chr(10).join([f"- {k}: {v}" for k, v in example_table_3.items()])}
    Text:
    {example_text_3}
    Relevant entity: Glasair Aviation (may appear as subject or object)
    Explanation:
    - "Owner":
        - Table: Lists Jilin Hanxing Group as the owner
        - Text: Confirms "the company was sold to the Jilin Hanxing Group"
        - Reasoning → Past tense "was sold to" indicates acquisition → acquired_by
    - "Products":
        - Table: Lists homebuilt aircraft kits
        - Text: Mentions production
        - Reasoning → Attribute info only → None
    - "Industry", "Predecessor", "Founded", "Founders", "Headquarters", "Key people", "Website":
        - Table: Attribute info, not relationships
        - Text: Confirms background/history but no relational info
        - Reasoning → No relational predicate → None
    Predicate mapping:
    {json.dumps(example_mapping_3, indent=2)}

    --------------------------------------------------
    Example 4 (LocusPoint Networks)
    Table columns:
    {chr(10).join([f"- {k}: {v}" for k, v in example_table_4.items()])}
    Text:
    {example_text_4}
    Relevant entity: LocusPoint Networks (may appear as subject or object)
    Explanation:
    - "Owner":
        - Table: Lists Blackstone Group (99%) as owner
        - Text: Confirms "the company is 99% owned by The Blackstone Group"
        - Reasoning → Present tense "is owned by" indicates current ownership → subsidiary_of

    - "Company type", "Industry", "Founded", "Defunct", "Headquarters", "Key people", "Website":
        - Table: Attribute information only
        - Text: Confirms history/background, but does not indicate any relational action or external entity
        - Reasoning → These are intrinsic attributes of the company, not relationships → None
    Predicate mapping:
    {json.dumps(example_mapping_4, indent=2)}

    --------------------------------------------------
    Example 5 (Sony Pictures Television Kids)
    Table columns:
    {chr(10).join([f"- {k}: {v}" for k, v in example_table_5.items()])}
    Text:
    {example_text_5}
    Relevant entity: Sony Pictures Television Kids (may appear as subject or object)
    Explanation:
    - "Title":
        - Table: Lists Hilda and the Mountain King as a production
        - Text: Does not directly mention this title but context implies productions by the company
        - Reasoning → The title represents a produced work → product_or_service_of
    - "Network":
        - Table: Lists Netflix as the network
        - Text: No explicit mention, but network implies distribution/platform relationship
        - Reasoning → Network distributes or hosts the content → client_of
    - "Co-production with":
        - Table: Lists Mercury Filmworks and Nobrow Press
        - Text: Not explicitly mentioned, but “co-production” clearly indicates joint work
        - Reasoning → Joint production activity → collaboration
    - "Released":
        - Table: Lists release year
        - Text: No relational information
        - Reasoning → Attribute only → None
    Predicate mapping:
    {json.dumps(example_mapping_5, indent=2)}

    --------------------------------------------------
    Example 6 (SES Americom)
    Table columns:
    {chr(10).join([f"- {k}: {v}" for k, v in example_table_6.items()])}
    Text:
    {example_text_6}
    Relevant entity: "SES Americom (may appear as subject or object)
    Explanation:
    - "Satellite":
    - Table: Lists AMC-1 through AMC-21 as satellite names.
    - Text: Explicitly mentions SES acquired GE Americom and renamed satellites.
    - Reasoning → These satellites are assets owned/acquired by SES Americom → acquired_by.
    - Manufacturer", "Position", "Model", "Launched", "Launch vehicle", "Comments"
    - Table: Lists technical, operational, or descriptive details of each satellite.
    - Text: Provides additional information about builders or operational history but does not indicate ownership, usage, or client relationships.
    - Reasoning → All these columns describe attributes of the satellites, not relational predicates → None.
    Predicate mapping:
    {json.dumps(example_mapping_6, indent=2)}

    --------------------------------------------------
    New Table - {new_title}
    Table columns:
    {chr(10).join([f"- {k}: {v}" for k, v in new_table.items()])}
    Text:
    {new_text or "No additional text provided."}
    Relevant entity: {new_title} (may appear as subject or object)

    Instructions for new table:
    - Map each column to one of the allowed predicates.
    - Columns with no matching predicate → None.
    - Use text to justify the mapping only when it clarifies or adds information. If the text is missing or ambiguous, it is safe to rely on table alone.
    - The entity may appear as subject or object in the triple.
    - Only use allowed predicates.
    - Return JSON only.
    """
    # Send request to OpenAI ChatCompletion API
    llm_answer = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=512,
        temperature=0,
    )

    raw_output = llm_answer.choices[0].message.content
  
    return safe_json_extract(raw_output)

# ============================================================
# FULL PIPELINE
# ============================================================
def run_system(input_json_path):
    
    print(f"\nLoading file: {input_json_path}")

    with open(input_json_path, "r", encoding="utf-8") as f:
        data_json = json.load(f)

    rows = []

    # --------------------------------------------------------
    # LOOP THROUGH INPUT DATA
    # --------------------------------------------------------

    for doc_id, doc_entry in tqdm(list(data_json.items()) if isinstance(data_json, dict) else enumerate(data_json)):

        new_text = "\n".join(doc_entry.get("text", []))
        title = doc_entry.get("title", "")
        cleaned_title = clean_title(title)
        title_entity = cleaned_title

        for table_idx, table in enumerate(doc_entry.get("tables", [])):
            new_table = {}
            for col, vals in table.items():
                new_table[col] = vals

            # ====================================================
            # STAGE 1: PREDICATE MAPPING
            # ====================================================
            try:
                mapping = get_mapping(new_table, new_text, cleaned_title)
            except Exception as e:
                print("Mapping error:", e)
                continue

            # ====================================================
            # STAGE 2: TRIPLE GENERATION
            # ====================================================
            for column, values in new_table.items():

                predicate = mapping.get(column)
                if not predicate:
                    continue

                if not isinstance(values, list):
                    values = [values]

                for value in values:

                    entities = extract_entities(str(value))

                    for val in entities:

                        if numeric_pattern.match(val):
                            continue

                        col_clean = column.lower().strip()

                        if col_clean not in reverse_columns:
                            subject = title_entity
                            obj = val
                        else:
                            subject = val
                            obj = title_entity

                        rows.append({
                            "document_id": doc_id,
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj
                        })
            df = pd.DataFrame(rows)

            if not df.empty:

                df = df[["document_id", "subject", "predicate", "object"]]

                df.drop_duplicates(
                    subset=["document_id", "subject", "predicate", "object"],
                    inplace=True
                )
                print("\n================ FINAL RDF TRIPLES ================\n")
                print(df.to_string(index=False))

                print(f"\nTotal triples generated: {len(df)}")

            else:
                print("\n No triples generated for this table.")
            
            time.sleep(20)

    return rows


if __name__ == "__main__":

    input_path = "cleaned_data/business_data.json" 

    run_system(input_path)