def a(table_name):
    print(table_name)


# a('table_name')
# a(['data_sample_mk', 'data_schema', 'table_name'])
from jinja2 import Template


def query1(table_names, question: str = '员工表中有多少条记录'):
    data_samples = []
    data_sample_mks = []
    data_schemas = []
    for table_name in table_names:

        if table_name in parser.table_names:

            # 表样例
            data_sample = parser.get_table_sample(table_name)
            data_sample_mk = data_sample.to_markdown()

            # 表格式
            data_schema = parser.get_table_fields(table_name).to_markdown()
            data_samples.append(data_sample)
            data_sample_mks.append(data_sample_mk)
            data_schemas.append(data_schema)
        else:
            import sqlite3
            conn = sqlite3.connect('chinook.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            tables = pd.DataFrame([tables])
            # data_samples.append(tables.to_markdown())
            data_samples.append('')
            data_sample_mks.append('')
            data_schemas.append('')
    try:
        print('question', question)
        # 生成答案SQL
        # input_str = answer_prompt.format(table_name=table_name, data_sample_mk=data_sample_mk, data_schema=data_schema, question=question)

        template = Template(answer_prompts)
        input_str = template.render(table_name=table_names, data_sample_mk=data_sample_mks, data_schema=data_schemas,
                                    question=question)
        print('提问的提示词', input_str)
        answer = ask_glm2(input_str, nretry=1)['choices'][0]['message']['content']
        # print('answer', answer)
        answer = answer.strip('`').strip('\n').replace('sql\n', '')
        # print('answer2', answer)
        # 判断SQL是否符合逻辑
        flag, _ = parser.check_sql(answer)
        # print('flag', flag)
        if not flag:
            raise Exception('error check_sql')
        # 获取SQL答案
        sql_answer = parser.execute_sql(answer)
        if len(sql_answer) > 1:
            raise Exception('error check_sql')
        sql_answer = sql_answer[0]
        sql_answer = ' '.join([str(x) for x in sql_answer])
        # 将提问改写，更加符合用户风格
        input_str = nl_answer_prompt.format(question=question, answer=sql_answer)
        # print('将提问改写，更加符合用户风格input_str', input_str)
        question = ask_glm2(input_str)['choices'][0]['message']['content']
        print('nl_answer：', question)
    except Exception as e:
        print('error', e)
    # query1(['employees'],'员工表中有多少条记录')


# query1(['employees','customers'],'在数据库中所有客户个数和员工个数分别是多少')
query1(['sqlite_master'], '数据库中总共有多少张表')