from dotenv import load_dotenv

from utils import init_args
from runnable import Runnable
from open_ai import OpenAIChatModel
from table_extraction import UnstructuredTableExtractor
from prompts import system_prompt, human_prompt
from markdown_remover import MarkdownRemover
from tqdm import tqdm

import os
import csv
import json

load_dotenv()

if __name__ == "__main__":
    args = init_args()
    r = Runnable(args)

    if len(args["load_query_from_file"]) > 0:
      md_remover = MarkdownRemover()
      openai_model = OpenAIChatModel(os.environ["OPENAI_MODEL_NAME"], float(os.environ["OPENAI_TEMPERATURE"]))
      with open(args["load_query_from_file"], 'r') as file:
        data = json.load(file)
      
      if os.path.isdir(args["pdf"]):
        file_names = os.listdir(args["pdf"])
      elif os.path.isfile(args["pdf"]):
        file_names = [args["pdf"]]
      else:
        raise ValueError(f"wrong file name")
  
      for file_name in file_names:
        splitted_file_name = file_name.split(".")
        if splitted_file_name[-1] != "pdf":
          continue
        
        dir_name = '.'.join(splitted_file_name[:-1])

        args["pdf"] = file_name
        gri_code_to_page = {}
        tables_as_html = set()

        for gri_code, description in data.items()[:3]:
          if gri_code not in gri_code_to_page.keys():
            gri_code_to_page[gri_code] = []

          args["query"] = description
          r.set_args(args)
          s = r.run()

          ute = UnstructuredTableExtractor("yolox", "hi_res")

          for doc in tqdm(s[:20]): #keep only the top-20 rated pages
            tables = ute.extract_table_unstructured([doc]) #extract tables

            for table in tables:
              for i in range(len(table)):
                tables_as_html.add((table[i].metadata.text_as_html, doc.page_content, doc["page"], i))
                gri_code_to_page[gri_code].append((doc["page"], i))

        openai_results = []
        for table_html in tables_as_html:

          res = openai_model.invoke(
              system_prompt,
              human_prompt.format(table_html[0], table_html[1])
          )
          #print(res.content)

          if not os.path.exists(dir_name):
            os.mkdir(f"table_dataset/{dir_name}")

          content = md_remover.unmark(res.content)

          with open(f'table_dataset/{dir_name}/{str(table_html[-2])}_{str(table_html[-1])}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([content])

        with open(f'table_dataset/{dir_name}/metadata.json', 'w') as json_file:
          json.dump(gri_code_to_page, json_file, indent=4)
          
    else:
      s = r.run()
