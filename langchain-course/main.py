from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama



load_dotenv()


def main():
    print("Hello from langchain-course!")
    
    information = '''
    Basebol[1] ou beisebol[2][3] (do inglês baseball), é um desporto de taco e bola praticado por duas equipes de nove jogadores, que alternadamente ocupam as posições de ataque e defesa.

    O objetivo é pontuar batendo com um bastão em uma bola lançada e depois correr pelas quatro bases do campo. Um jogador da equipe atacante pode parar em uma das bases e, depois, avançar com a ajuda da rebatida de um companheiro. Os times trocam de posição assim que três rebatedores são eliminados. Um turno de ataque e defesa de cada time representa uma entrada e nove entradas compõem um jogo profissional. O time com mais corridas no final vence.

    É um desporto muito popular na América do Norte, em alguns países da América Central, no Caribe hispanófono e no Extremo Oriente. Nos Estados Unidos a modalidade é a que atrai mais espectadores aos estádios. O basebol, apresentado a título de modalidade de demonstração em vários Jogos Olímpicos dispersos ao longo do tempo, foi incluído no programa oficial dos Jogos Olímpicos de Barcelona em 1992, sendo posteriormente removido a partir de Londres 2012.

    No entanto, em 2016, o Comitê Olímpico Internacional aprovou o beisebol como modalidade para os Jogos Olímpicos de Verão de 2020.[4] Apesar de não ter sido incluído nos Jogos de 2024 em Paris [5], o beisebol retornará como esporte olímpico em 2028, para as Olimpíadas de Los Angeles, nos Estados Unidos.[6]
    '''

    summary_template = '''
    Given the information {information} about a sport I want to create:
    1. A short summary
    2. One insteresting fact about it
    '''


    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )


    #llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    llm = ChatOllama(model="qwen3:1.7b", temperature=0)
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information":information})
    print(response.content)

if __name__ == "__main__":
    main()
