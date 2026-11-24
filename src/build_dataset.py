"""
Dataset Generator & Preprocessor for Bank Sense Disambiguation
Generates 2,400+ balanced, realistic, unique sentences containing the word 'bank'
in both Financial and River contexts.
"""

import os
import re
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
random.seed(42)

# --- 1. CORE SEED SAMPLES (Linguistically diverse sentences) ---
FINANCIAL_SEEDS = [
    "I deposited my monthly salary directly into the bank account.",
    "The fisherman went to the local bank to deposit his weekly earnings.", # Contrast sentence!
    "The bank approved our commercial mortgage application this morning.",
    "Customers lined up in front of the bank waiting for the doors to open.",
    "The central bank announced an unexpected hike in interest rates yesterday.",
    "She checked her online bank balance before authorizing the transaction.",
    "The commercial bank offers favorable savings account interest rates.",
    "He withdrew three hundred dollars from the bank ATM machine.",
    "The investment bank reported record profits in the fourth quarter.",
    "The teller smiled warmly as she handed me the bank receipt.",
    "Our company secured a multi-million dollar loan from a private bank.",
    "The bank manager explained the terms of the fixed deposit investment.",
    "She keeps her heirloom jewelry locked securely in a bank safe deposit box.",
    "The national bank intervened to stabilize the falling exchange rate.",
    "They transferred funds electronically between the two bank institutions.",
    "Due to insolvency, the regional bank was taken over by federal regulators.",
    "The bank charges an overdraft penalty when account balances fall below zero.",
    "We visited the downtown bank branch to open a new joint checking account.",
    "The credit card company is affiliated with a major international bank.",
    "A sophisticated cyber attack targeted the core servers of the foreign bank."
]

RIVER_SEEDS = [
    "The fisherman sat quietly beside the muddy bank of the river.",
    "The banker took a peaceful walk along the grassy bank of the river.", # Contrast sentence!
    "Wild weeping willows leaned gently over the calm river bank.",
    "Children were skipping smooth stones from the sandy bank across the stream.",
    "Heavy monsoon rains caused the raging river to overflow its steep bank.",
    "A solitary blue heron stood motionless along the gravelly bank watching for fish.",
    "We pitched our camping tent on a dry elevated section of the river bank.",
    "Rafting enthusiasts pulled their inflatable boats onto the rocky bank.",
    "Soil erosion has severely damaged the southern bank of the winding river.",
    "Small freshwater crabs scuttled into burrows along the muddy bank.",
    "The scenic trail winds gracefully alongside the green bank of the mountain stream.",
    "Thick reeds and aquatic grasses grow profusely along the moist river bank.",
    "The river burst its left bank after three days of relentless downpours.",
    "Local volunteers gathered to clean up trash along the polluted river bank.",
    "An old wooden canoe was tied to a sturdy willow on the river bank.",
    "We enjoyed a sunset picnic sitting right on the grassy bank listening to water flow.",
    "The rapid water current eroded several feet of the soft clay bank.",
    "Otters built their underground dens burrowed into the steep river bank.",
    "Fishermen set up their rods and folding chairs along the sloping river bank.",
    "Sunlight glittered on the water surface near the quiet wooded bank."
]

# --- 2. COMBINATORIAL PATTERNS FOR SCALED BALANCED DATASET ---

# Financial components
FIN_SUBJECTS = [
    "The accountant", "The financial analyst", "The customer", "The small business owner",
    "The investor", "The entrepreneur", "The branch manager", "The senior teller",
    "The corporate auditor", "The loan applicant", "The credit officer", "The young graduate",
    "The family", "The elderly pensioner", "The freelance developer", "The retail merchant",
    "The real estate agent", "The mortgage broker", "The startup founder", "The treasurer",
    "Our company", "The international client", "The debt collector", "The financial adviser"
]

FIN_ACTIONS = [
    ("deposited savings at the", "to earn compound interest on the capital"),
    ("withdrew cash from the", "to pay for emergency medical expenses"),
    ("applied for a business loan at the", "with competitive interest rates"),
    ("opened a premium checking account with the", "providing complimentary checks"),
    ("negotiated credit terms with the", "to expand regional warehouse facilities"),
    ("verified account credentials at the", "prior to wire authorization"),
    ("disputed an unauthorized charge through the", "fraud prevention department"),
    ("transferred international currency via the", "using direct SWIFT payment"),
    ("received formal mortgage approval from the", "after rigorous financial scrutiny"),
    ("scheduled a consultation at the", "to discuss retirement portfolio options"),
    ("consolidated student debt through the", "at a reduced fixed rate"),
    ("safeguarded legal title deeds inside the", "private vault facility"),
    ("queried the annual percentage yield with the", "customer service representative"),
    ("reviewed the monthly statement issued by the", "for billing discrepancies"),
    ("settled the outstanding overdraft balance with the", "before the penalty cycle"),
    ("submitted certified financial affidavits to the", "underwriting committee"),
    ("refinanced the commercial property with the", "saving thousands in interest"),
    ("authorized an automated monthly debit at the", "for utility bill payments"),
    ("requested an emergency cashier check from the", "for purchasing real estate"),
    ("inspected investment certificates at the", "during the quarterly review")
]

FIN_VARIANTS = [
    "downtown commercial bank", "local savings bank", "licensed banking institution",
    "national central bank", "trusted retail bank", "regional cooperative bank",
    "merchant investment bank", "offshore financial bank", "chartered bank",
    "community credit bank", "city center bank", "multinational private bank"
]

# River components
RIV_SUBJECTS = [
    "The experienced angler", "The nature photographer", "The wildlife biologist",
    "The energetic camper", "The weekend hiker", "The field botanist",
    "The local canoeist", "The bird watcher", "The weary traveler",
    "The landscape artist", "The river guide", "The forestry officer",
    "A group of schoolchildren", "A flock of migratory ducks", "A patient fisherman",
    "The environmental surveyor", "The outdoor adventurer", "The park ranger",
    "The wandering poet", "The young scout troop", "A family of wild otters",
    "The kayak instructor", "The village elder", "The conservation volunteer"
]

RIV_ACTIONS = [
    ("rested quietly on the grassy", "watching ripples expand across the clear water"),
    ("spotted a nesting kingfisher along the", "perched on an overhanging branch"),
    ("set up a sturdy tripod on the muddy", "to capture the golden hour reflection"),
    ("walked slowly down the sloping", "carefully stepping over slippery rocks"),
    ("tied the aluminum rowboat securely to the", "using a weather-resistant rope"),
    ("pitched an expedition tent above the", "to avoid nocturnal water surges"),
    ("collected aquatic soil samples from the", "for ecological contamination testing"),
    ("sat peacefully reading a book along the", "enjoying the gentle river breeze"),
    ("observed seasonal flood erosion along the", "where roots were exposed to the current"),
    ("skipped flat stones across the stream from the", "competing for the most bounces"),
    ("cleared plastic debris from the scenic", "to protect downstream marine habitats"),
    ("sketched the weeping willows that framed the", "in charcoal and watercolors"),
    ("cast a dry fly lure into the eddy from the", "hoping to hook a rainbow trout"),
    ("noticed animal footprints pressed into the", "leading straight toward the shallows"),
    ("listened to the rushing water beside the", "as twilight settled over the valley"),
    ("monitored river current speed along the", "recording hydrologic data on a tablet"),
    ("found freshwater mussel shells scattered on the", "left behind by feeding wildlife"),
    ("built a small campfire safely away from the", "underneath towering pine trees"),
    ("photographed blooming wildflowers covering the", "during peak spring blossom"),
    ("navigated slippery moss-covered stones along the", "to reach a quiet fishing hole")
]

RIV_VARIANTS = [
    "river bank", "stream bank", "rocky creek bank", "grassy river bank",
    "sandy river bank", "muddy water bank", "winding brook bank",
    "overgrown river bank", "forested river bank", "sloping river bank",
    "shady river bank", "peaceful watercourse bank"
]


def generate_dataset():
    data = []
    
    # Add seeds
    for s in FINANCIAL_SEEDS:
        data.append({"sentence": s.strip(), "label": "financial"})
    for s in RIVER_SEEDS:
        data.append({"sentence": s.strip(), "label": "river"})
        
    # Generate Financial Sentences
    fin_generated = set(s["sentence"] for s in data if s["label"] == "financial")
    
    stop_fin = False
    for subj in FIN_SUBJECTS:
        if stop_fin: break
        for action, desc in FIN_ACTIONS:
            if stop_fin: break
            for variant in FIN_VARIANTS:
                if len(fin_generated) >= 1400:
                    stop_fin = True
                    break
                # Template 1: Direct statement
                sent1 = f"{subj} {action} {variant} {desc}."
                if sent1 not in fin_generated and "bank" in sent1.lower():
                    fin_generated.add(sent1)
                    data.append({"sentence": sent1, "label": "financial"})
                    
                # Template 2: Prepositional fronting
                sent2 = f"At the {variant}, {subj.lower()} {action.split(' the')[0]} {desc}."
                if sent2 not in fin_generated and "bank" in sent2.lower():
                    fin_generated.add(sent2)
                    data.append({"sentence": sent2, "label": "financial"})

                # Template 3: Subordinate clause
                sent3 = f"When the transaction was processed, {subj.lower()} contacted the {variant} {desc}."
                if sent3 not in fin_generated and "bank" in sent3.lower():
                    fin_generated.add(sent3)
                    data.append({"sentence": sent3, "label": "financial"})

                # Template 4: Passive construction
                sent4 = f"Funds were officially transferred to the {variant} by {subj.lower()} {desc}."
                if sent4 not in fin_generated and "bank" in sent4.lower():
                    fin_generated.add(sent4)
                    data.append({"sentence": sent4, "label": "financial"})
                    
    # Generate River Sentences
    riv_generated = set(s["sentence"] for s in data if s["label"] == "river")
    
    stop_riv = False
    for subj in RIV_SUBJECTS:
        if stop_riv: break
        for action, desc in RIV_ACTIONS:
            if stop_riv: break
            for variant in RIV_VARIANTS:
                if len(riv_generated) >= 1400:
                    stop_riv = True
                    break
                # Template 1: Direct statement
                sent1 = f"{subj} {action} {variant} {desc}."
                if sent1 not in riv_generated and "bank" in sent1.lower():
                    riv_generated.add(sent1)
                    data.append({"sentence": sent1, "label": "river"})
                    
                # Template 2: Prepositional fronting
                sent2 = f"Along the {variant}, {subj.lower()} {action.split(' the')[0]} {desc}."
                if sent2 not in riv_generated and "bank" in sent2.lower():
                    riv_generated.add(sent2)
                    data.append({"sentence": sent2, "label": "river"})

                # Template 3: Temporal clause
                sent3 = f"As the morning mist cleared, {subj.lower()} approached the {variant} {desc}."
                if sent3 not in riv_generated and "bank" in sent3.lower():
                    riv_generated.add(sent3)
                    data.append({"sentence": sent3, "label": "river"})

                # Template 4: Descriptive setting
                sent4 = f"Near the verdant {variant}, {subj.lower()} {action.split(' the')[0]} {desc}."
                if sent4 not in riv_generated and "bank" in sent4.lower():
                    riv_generated.add(sent4)
                    data.append({"sentence": sent4, "label": "river"})

    df = pd.DataFrame(data)
    
    # Shuffle and clean
    df["sentence"] = df["sentence"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["sentence"]).reset_index(drop=True)
    
    # Filter strictly sentences that contain 'bank'
    df = df[df["sentence"].str.contains(r"\bbanks?\b", case=False, regex=True)].reset_index(drop=True)
    
    # Take balanced 1,050 samples per class (2,100 total, >= 1,000 per class required)
    n_per_class = min(len(df[df["label"] == "financial"]), len(df[df["label"] == "river"]), 1050)
    print(f"Sampling {n_per_class} per class...")
    fin_df = df[df["label"] == "financial"].sample(n=n_per_class, random_state=42, replace=False)
    riv_df = df[df["label"] == "river"].sample(n=n_per_class, random_state=42, replace=False)
    balanced_df = pd.concat([fin_df, riv_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    return balanced_df


if __name__ == "__main__":
    os.makedirs("c:/Users/hp/Downloads/Shivani Gen Ai/Bank_Sense_AI/data", exist_ok=True)
    df = generate_dataset()
    csv_path = "c:/Users/hp/Downloads/Shivani Gen Ai/Bank_Sense_AI/data/bank_sense_dataset.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Total dataset generated: {len(df)} records")
    print("Class distribution:")
    print(df["label"].value_counts())
    
    # Generate Stratified Splits: 70% Train, 15% Val, 15% Test
    train_df, test_val_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        test_val_df, test_size=0.50, random_state=42, stratify=test_val_df["label"]
    )
    
    train_df.to_csv("c:/Users/hp/Downloads/Shivani Gen Ai/Bank_Sense_AI/data/train.csv", index=False)
    val_df.to_csv("c:/Users/hp/Downloads/Shivani Gen Ai/Bank_Sense_AI/data/val.csv", index=False)
    test_df.to_csv("c:/Users/hp/Downloads/Shivani Gen Ai/Bank_Sense_AI/data/test.csv", index=False)
    
    print(f"Stratified Splits created:")
    print(f"Train: {len(train_df)} ({train_df['label'].value_counts().to_dict()})")
    print(f"Val:   {len(val_df)} ({val_df['label'].value_counts().to_dict()})")
    print(f"Test:  {len(test_df)} ({test_df['label'].value_counts().to_dict()})")
