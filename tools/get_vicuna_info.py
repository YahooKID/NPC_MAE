"""
Use FastChat with Hugging Face generation APIs.

Usage:
python3 -m fastchat.serve.huggingface_api --model lmsys/vicuna-7b-v1.3
python3 -m fastchat.serve.huggingface_api --model lmsys/fastchat-t5-3b-v1.0

!!! ATTENTION !!!
As our experience, the prompt will directly decide the final effect of your classification model!
pre-knowledge is necessary!
If you dataset is not a large one, you can try to make vicuna output a shorter one output with a suited prompt!
###
"""
import argparse

import torch
import os

from fastchat.model import load_model, get_conversation_template, add_model_args


@torch.inference_mode()
def main(args):
    # Load model
    model, tokenizer = load_model(
        "vicuna-7b",
        device="cuda",
        num_gpus=4,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=True,
        cpu_offloading=False,
        revision=args.revision,
        debug=args.debug,
    )
    vil = os.listdir("handle_npc_data/info")
    
    for file in vil:
        rfile = os.path.join("handle_npc_data/info", file)
        with open(rfile, "r") as f:
            txt = f.read()

        # Build the prompt with a conversation template
        msg = "Here is some base information of Nasopharyngeal carcinoma. Nasopharyngeal carcinoma (NPC), or nasopharynx cancer, is the most common cancer originating in the nasopharynx, most commonly in the postero-lateral nasopharynx or pharyngeal recess (fossa of Rosenmüller), accounting for 50 percent of cases. NPC occurs in children and adults. NPC differs significantly from other cancers of the head and neck in its occurrence, causes, clinical behavior, and treatment. It is vastly more common in certain regions of East Asia and Africa than elsewhere, with viral, dietary and genetic factors implicated in its causation. It is most common in males. It is a squamous cell carcinoma of an undifferentiated type. Squamous epithelial cells are a flat type of cell found in the skin and the membranes that line some body cavities. Undifferentiated cells are cells that do not have their mature features or functions. NPC may present as a lump or a mass on both sides towards the back of the neck. These lumps usually are not tender or painful but appear as a result of the metastatic spread of the cancer to the lymph nodes, thus causing the lymph nodes to swell. Lymph nodes are defined as glands that function as part of the immune system and can be found throughout the body. Swelling of the lymph nodes in the neck is the initial presentation in many people, and the diagnosis of NPC is often made by lymph node biopsy. Signs of nasopharyngeal cancer may appear as headaches, a sore throat, and trouble hearing, breathing, or speaking. Additional symptoms of NPC include facial pain or numbness, blurred or double vision, trouble opening the mouth, or recurring ear infections. If the ear infection does not present with an upper respiratory tract infection, then an examination should be done on the nasopharynx. This is due to the fact that, in adults, ear infections are less common than in children. Signs and symptoms related to the primary tumor include trismus, pain, otitis media, nasal regurgitation due to paresis (loss of or impaired movement) of the soft palate, hearing loss and cranial nerve palsy (paralysis). Larger growths may produce nasal obstruction or bleeding and a 'nasal twang'. Metastatic spread may result in bone pain or organ dysfunction. Rarely, a paraneoplastic syndrome of osteoarthropathy (diseases of joints and bones) may occur with widespread disease. NPC is caused by a combination of factors: viral, environmental influences, and heredity. The viral influence is associated with infection with Epstein–Barr virus (EBV). The Epstein-Barr virus is one of the most common viruses. 95 percent of all people in the U.S. are exposed to this virus by the time they are 30–40 years old. The World Health Organization does not have set preventative measures for this virus because it is so easily spread and is worldwide. Very rarely does Epstein-Barr virus lead to cancer, which suggests a variety of influencing factors. Other likely causes include genetic susceptibility and consumption of food (in particular salted fish) containing carcinogenic volatile nitrosamines. Various mutations that activate NF-κB signalling have been reported in almost half of NPC cases investigated.The association between Epstein-Barr virus and nasopharyngeal carcinoma is unequivocal in World Health Organization (WHO) types II and III tumors but less well-established for WHO type I (WHO-I) NPC, where preliminary evaluation has suggested that human papillomavirus (HPV) may be associated. EBV DNA was detectable in the blood plasma samples of 96 percent of patients with non-keratinizing NPC, compared with only 7 percent in controls. The detection of nuclear antigen associated with Epstein-Barr virus (EBNA) and viral DNA in NPC type 2 and 3, has revealed that EBV can infect epithelial cells and is associated with their transformation. The cause of NPC (particularly the endemic form) seems to follow a multi-step process, in which EBV, ethnic background, and environmental carcinogens all seem to play an important role. More importantly, EBV DNA levels appear to correlate with treatment response and may predict disease recurrence, suggesting that they may be an independent indicator of prognosis. The mechanism by which EBV alters nasopharyngeal cells is being elucidated to provide a rational therapeutic target.It is also being investigated as to whether or not chronic sinusitis could be a potential cause of cancer of the nasopharynx. It is hypothesised that this may happen in a way similar to how chronic inflammatory conditions in other parts of the body, such as esophagitis sometimes leading to Barrett's esophagus because of conditions such as gastroesophageal reflux disease. Risk factors: Nasopharyngeal carcinoma, classified as a squamous cell cancer, has not been linked to excessive use of tobacco. However, there are certain risk factors that can predispose an individual to NPC if exposed to them. These risk factors include: having Chinese, or Asian, ancestry, exposure to Epstein- Barr virus (EBV), unknown factors that result in rare familial clusters, and heavy alcohol consumption. Epstein- Barr virus infects and persists in more than 90 percent of world population. Transmission of this virus occurs through saliva and is more commonly seen in developing countries where there are living areas are more packed together and less hygienic. Replication of this virus can occur in the oropharyngeal epithelial tissue and nasopharyngeal tissue. EBV primarily targets B lymphocytes. Patients diagnosed with NPC were found to show elevated levels of the antibodies against the EBV antigen than in individuals not diagnosed with NPC. Individuals that are exposed to cigarette smoking have an increased risk of developing NPC by 2- to 6-fold. Approximately two-thirds of patients with type 1 NPC was attributed to smoking in the United States. However the declining rates of smoking in the US can be associated with less prevalence of type 1 NPC. In southern China and North Africa, it has been suggested that high smoking rates come from wood fires in the country rather than cigarette smoking. NPC can be treated by surgery, by chemotherapy, or by radiotherapy. There are different forms of radiation therapy, including 3D conformal radiation therapy, intensity-modulated radiation therapy, particle beam therapy and brachytherapy, which are commonly used in the treatments of cancers of the head and neck. The expression of EBV latent proteins within undifferentiated nasopharyngeal carcinoma can be potentially exploited for immune-based therapies. Generally, there are three different types or treatment methods that can be used for patients with nasopharyngeal carcinoma. These three treatments are radiation therapy, chemotherapy, and surgery. Although there are currently three treatment methods, there are clinical trials taking place that may develop more effective treatments for NPC. A clinical trial is research study that works to develop new treatment techniques or to gain more information about or improve current methods. If an effective treatment comes out of the clinical trial, then this method may become a new standard treatment method. During the course of, or following, treatment, tests may be done in order to determine if the treatment is working, or if treatment needs to be dropped or changed. Tests that are done after treatment to determine the condition of patient after completing treatment are called follow-up tests and tell the doctor if the patients condition has changed or if the cancer has come back. Radiation therapy uses high energy x-rays or other types of radiation aimed to prevent cancer cells from growing or kill them altogether. This kind of therapy can be administered to the patient externally or internally. With external radiation, a machine is used to send targeted radiation to the cancer site. A mesh mask is used on the patient in order to keep their head and neck still while the machine rotates to send out beams of radiation. In undergoing this kind of treatment, healthy cells may also be damaged during the process. Therefore, there are 2 other forms of radiation therapy that decreases the likelihood of damaging nearby healthy cells: intensity-modulated radiation therapy and stereotactic radiation therapy. Intensity-modulated radiation therapy (IMRT) uses 3D images of the size and shape of the tumor to then direct thin beams of radiation at different intensities from multiple angles. In stereotactic radiation therapy, radiation is aimed directly at the tumor. In this therapy, the total amount of radiation is divided into smaller doses that will be given over the course of several days. Using radiation therapy as a cancer treatment method depends on the type and stage of cancer; however, internal and external radiation therapies can be used to treat NPC. If external radiation therapies are being aimed at the thyroid, then this could affect the way the thyroid works. For that reason, blood tests are done before and after radiation to check thyroid hormone levels. Chemotherapy works as cancer treatment by using drugs that stop the growth of cancer cells, by either killing the cells or preventing them from dividing. This kind of therapy can be administered systemically or regionally. Systemic chemotherapy is when the chemotherapy is taken orally or is injected into a vein or muscle. In this method, the drug circulates through the blood system and can reach cancer cells throughout the body. Regional chemotherapy is when chemotherapy is administered directly into the cerebrospinal fluid, an organ, or a body cavity, for example, the abdomen. In this way, the drugs will mainly affect cancer cells in that area. However, the type of chemotherapy that is administered to the patient depends on the type and stage of the cancer. Additionally, chemotherapy can be used as an adjuvant therapy after radiation to lower the risk of recurrence in the patient. If given after radiation, chemotherapy can be used to kill any cancer cells that may have remained. Surgery can be used as a method to determine whether there is cancer present or to remove cancer from the body. If the tumor does not respond to radiation therapy, then the patient may undergo an operation to have it removed. Cancers that may have spread to the lymph nodes may require the doctor to remove lymph nodes or other tissue in the neck. Now you need to review a Nasopharyngeal carcinoma patient. I will tell you some information of this patient, you should give me some review about this patient."
        #msg = args.message
        conv = get_conversation_template(args.model_path)
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], "I got this message and I'm a  nasopharyngeal cancer specialist.")
        conv.append_message(conv.roles[0], "Ok, I know now you are a nasopharyngeal cancer specialist. Whenever I send you related information about a nasopharyngeal cancer patient, you need to assess the difficulty level of the patient's recovery and the possible recovery time in a short word.")
        conv.append_message(conv.roles[1], "I get it, I will do it.")
        conv.append_message(conv.roles[0], "{} Briefly summarize the patient's condition and assess the difficulty of their treatment.".format(txt))
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Run inference
        inputs = tokenizer([prompt], return_tensors="pt").to(args.device)
        output_ids = model.generate(
            **inputs,
            do_sample=True if args.temperature > 1e-5 else False,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )

        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        outputs = tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

        # Print results
        print(f"{conv.roles[0]}: {msg}")
        print(f"{conv.roles[1]}: {outputs}")
        with open(os.path.join("handle_npc_data/vicuna_res", file), "w+") as f:
            f.write(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--message", type=str, default="Hello! Who are you?")
    args = parser.parse_args()

    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    main(args)
