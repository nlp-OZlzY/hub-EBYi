
"""
RAG 文档问答演示 - 本地数据版本
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.embedding_service import EmbeddingService

class DemoQA:
    def __init__(self):
        self.embed = EmbeddingService()
        self.documents = []
        self.vectors = []
        
    def load_documents(self):
        """Load test documents"""
        test_docs = [
            {"text": "汽车发动机是汽车的核心部件，负责产生动力驱动车辆前进。发动机通过燃烧燃料将化学能转化为机械能。", "page": 1, "file": "汽车知识手册.pdf"},
            {"text": "汽车保养周期建议每5000公里进行一次常规保养，包括更换机油和机油滤清器。定期保养可以延长发动机寿命。", "page": 5, "file": "汽车知识手册.pdf"},
            {"text": "领克汽车客服热线是4006-010101，工作时间为周一至周日9:00-21:00。如有任何问题可以随时联系客服。", "page": 10, "file": "汽车知识手册.pdf"},
            {"text": "汽车底盘包括传动系统、行驶系统、转向系统和制动系统四个主要部分。", "page": 3, "file": "汽车知识手册.pdf"},
            {"text": "汽车电气系统包括蓄电池、发电机、启动机和各种用电设备。蓄电池负责储存电能供启动使用。", "page": 8, "file": "汽车知识手册.pdf"}
        ]
        
        print("Loading documents...")
        for doc in test_docs:
            vector = self.embed.embed(doc['text'])
            self.documents.append(doc)
            self.vectors.append(vector)
            print(f"  Loaded page {doc['page']}: {doc['text'][:30]}...")
        
        print(f"\nLoaded {len(self.documents)} documents")
    
    def search(self, query, limit=3):
        """Search for similar documents"""
        query_vector = self.embed.embed(query)
        
        # Calculate cosine similarity
        results = []
        for i, doc in enumerate(self.documents):
            # Simple dot product similarity
            similarity = sum(a * b for a, b in zip(query_vector, self.vectors[i]))
            results.append({
                "text": doc['text'],
                "page": doc['page'],
                "file": doc['file'],
                "similarity": similarity
            })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]
    
    def answer(self, query):
        """Answer a question"""
        results = self.search(query)
        
        if not results:
            return "没有找到相关信息。", []
        
        answer = f"根据检索到的信息，关于 \"{query}\" 的回答如下：\n\n"
        sources = []
        
        for i, result in enumerate(results):
            answer += f"【来源: {result['file']} 第{result['page']}页】\n{result['text']}\n\n"
            sources.append({
                "file": result['file'],
                "page": result['page'],
                "similarity": f"{result['similarity']:.2%}"
            })
        
        answer += f"（共检索到 {len(results)} 条相关记录）"
        return answer, sources

def main():
    parser = __import__('argparse').ArgumentParser(description="RAG Demo QA")
    parser.add_argument('query', nargs='?', help='查询问题')
    args = parser.parse_args()
    
    qa = DemoQA()
    qa.load_documents()
    
    if args.query:
        answer, sources = qa.answer(args.query)
        print("\n" + "="*60)
        print(answer)
        print("="*60)
        
        if sources:
            print("\n来源:")
            for i, source in enumerate(sources, 1):
                print(f"  {i}. {source['file']} (第{source['page']}页, 相似度: {source['similarity']})")
        
        # Save to file
        with open('demo_result.txt', 'w', encoding='utf-8') as f:
            f.write(answer)
        print("\n结果已保存到 demo_result.txt")
    else:
        print("\n" + "="*60)
        print("        RAG 文档问答演示")
        print("="*60)
        print("输入问题开始问答，输入 'exit' 退出")
        print("="*60)
        
        while True:
            try:
                query = input("\n请输入问题: ").strip()
                if query.lower() in ['exit', 'quit', 'q']:
                    print("再见！")
                    break
                if not query:
                    continue
                
                answer, sources = qa.answer(query)
                print("\n" + "="*60)
                print(answer)
                print("="*60)
                
            except KeyboardInterrupt:
                print("\n再见！")
                break

if __name__ == "__main__":
    main()
