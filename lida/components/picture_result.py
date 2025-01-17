from zhipuai import ZhipuAI
from lida.config import GLM_API_KEY
import base64

FORMAT_INSTRUCTIONS = """
请根据解读模板对图表进行解读，给出探索解读结果。模板如下：
###
图表解释：
	目的：分析[主要比较的变量]与[次要比较的变量]之间的关系。
	数据说明：图表中不同的柱子/点/折线等等代表什么，它们有什么关系，它们有什么特征
详细分析：
	深入探讨不同类别下的度量指标的表现。这包括比较各类别间的差异，分析特定类别中的趋势，以及探索这些差异或趋势可能的原因。目的是揭示主要和次要分类变量之间的相互作用和影响，从而提供对目标变量影响因素的深入理解。用一段话进行说明。
结论：
	根据标题为读者提供了一个清晰的总结，强调分析的主要发现，并指出这些发现对实际应用的潜在影响和重要性。用一段话进行总结。
###
"""

client = ZhipuAI(api_key=GLM_API_KEY)

def picture_result_generate(image_url):
    try:
        response = client.chat.completions.create(
        model="glm-4v-flash",
        messages=[
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": FORMAT_INSTRUCTIONS
            },
            {
                "type": "image_url",
                "image_url": {
                    "url" : image_url
                }
            }
            ]
        }
        ],
        top_p= 0.7,
        temperature= 0.95,
        max_tokens=1024
        )
        return response.choices[0].message
    except Exception as e:
        print(e)
    return None
    

# with open('a.png', 'rb') as image_file:  # 以二进制读取模式打开图片文件
#     image_data = image_file.read()  # 读取图片文件的内容
# image_base64 = base64.b64encode(image_data).decode('utf-8')
# image_url = f'data:image/jpeg;base64,{image_base64}'  # 创建一个包含图片 Base64 编码的数据 URL
# content = picture_result_generate(image_url).content
# print(content)