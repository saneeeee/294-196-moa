SYSTEM_PROMPT = """
You are a helpful assistant used for data processing. You have expert understanding of taking in
chunks of textual data and generating question answer pairs for supervised fine tuning. You do not incorporate your own knowledge
and only use factual information form the provided knowledge text."""

USER_PROMPT = """
I want to create question answer pairs for the following knowledge text, format the pairs like this example:
<example>
<example_knowledge_text>
1.035 Co-Owners (VC §§4150.5 and 9852.5)
A vehicle or vessel may be owned by two or more co-owners. Co-owner names may be joined by “and”, “and/or”, or “or”. All owners mustendorse the title or registration application to register the vehicle/vessel, but the requirements for releasing ownership vary. Refer to Chapter 11.
Certificates issued for applications not indicating “and” or “or” between the names will show “and” as represented by a slash (/) between the names.
* The signatures of all owners are required to transfer ownership when the co-owner names are joined by “and”. Ownership passes to the surviving co-owner upon the death of a co-owner or, with the surviving co-owner’s release, to a new owner. A deceased co-owner’s interest may only be released by one of the following:
   * Heir of the deceased with an Affidavit for Transfer Without Probate California Titled Vehicle or Vessels Only (REG 5) form.
   * Administrator with Letters of Administration.
   * Executor with Letters Testamentary.
* The signature of only one owner is required to transfer ownership when the co-owner names are joined by “and/or” or “or”. A surviving co-owner’s signature on the title releases all owners’ interest unless “Tenants in Common” or “COMPRO” follows the co-owner’s names.
* A REG 5 cannot be used to circumvent the interest of a surviving owner when the vehicles are jointly owned by two or more persons and one of the owners is deceased. However, the surviving owner (if they are the heir) may complete a REG 5 to release the interest of the deceased owner. The California Certificate of Title must be signed twice, once by surviving owner and once for the deceased owner countersigned by the heir. If owned jointly by two or more deceased owners, a REG 5 for the most recently deceased owner and a death certificate for each owner is required.
Tenants in Common—When “Tenants in Common” follows the names of co-owners, the interest of a deceased co-owner reverts to the deceased co-owner’s estate, not to the surviving co-owner. Ownership may be transferred with Letters Testamentary, Letters of Administration, or an Affidavit for Transfer Without Probate California Titled Vehicle or Vessels Only (REG 5) form.
COMPRO—When “COMPRO” (community property) follows the names of co-owners, ownership passes to the surviving owner after the deceased co-owner’s interest is executed by the:
* Heir of the deceased with a REG 5, if the estate was not probated.
* Administrator of the estate with Letters of Administration.
* Executor of the estate with Letters Testamentary.
Ownership may be transferred to a new owner with the surviving co-owner’s release.
JTRS—When “JTRS” (joint tenants with right of survivorship) follows the names of co-owners:
* All owners must release interest during the lifetime of the co-owners.
* Upon the death of an owner, interest is released by the surviving co-owner.
* A copy of the deceased owner’s death certificate must accompany the application.
* The signature of the surviving co-owner(s) on the title releases all owner interests.
TOD (Transfer on Death)—Refer to Chapter 11.
<end_example_knowledge_text>
<example_qa_pair>
[{{'role': 'user', 'content': 'Can I register a car to multiple users in California'}},
{{'role': 'assistant', 'content': 'Absolutely, you can register a car under multiple owners in California. The state allows co-ownership with different options like "and," "or," or "and/or" between names, which affects how ownership transfers work. All co-owners need to sign the initial registration, but transfer requirements vary. There are also special rules for situations like when an owner passes away or for specific ownership types like community property. The key is to choose the right co-ownership option that fits your situation, as it'll determine things like whether you need all owners to sign off on a sale or just one.'}}]
<end_example_qa_pair>
<end_example>

Generate questions that people will commonly ask about this the given knowledge base.
Make sure your answers to the questions are clear and comprehensive, the answers must simulate an expert clearly yet succintly provinding a factual answer.
Make your answers medium lenght, but do not omit important information.

Generate at leat {0} question answer pairs and cover all possible questions that someone might ask about this knowledge text, before genrating each question, say [n/{0}]
Where n is the pair number, make sure to follow the template provided in the example.

so outputs will follow the structure
[0/{0}]
[{{"role": "user", "content:" "..."}}, {{"role": "assistant", "content:" "..."}}]

[1/{0}]
[{{"role": "user", "content:" "..."}}, {{"role": "assistant", "content:" "..."}}]

...
<knowledge_text>
"""

def get_system_prompt(): 
    return SYSTEM_PROMPT


def get_user_prompt(scrape_file, n_questions=100):
    
    knowledge_base = ''
    with open(scrape_file, 'r') as f:
        knowledge_base += f.read() + "\n"
    return USER_PROMPT.format(n_questions) + knowledge_base + "\n<end_knowledge_text>"


    return 