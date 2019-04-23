# -*-Encoding:UTF-8-*-
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import json
import dateparser
import requests
import datetime
import happybase

#['info:corewords', 'info:url', 'info:join', 'info:post_user', 'info:uid', 'info:releasedate', 
#'info:reply', 'info:imgurl', 'info:title', 'info:longtype2', 'info:text', 'info:longtype1']

ES = Elasticsearch([{'host': '10.1.1.35', 'port': 9200}],
                   sniff_on_start=True, sniff_on_connection_fail=True, sniffer_timeout=200)
HBaseConnection = happybase.Connection('10.1.1.34', 9090)#34 9088
HBaseConnection.open()

#print(HBaseConnection.tables())
#exit()

WeiboHTable = HBaseConnection.table("weibosInfo")
WechatHTable = HBaseConnection.table("wechatInfo")
TiebaHTable = HBaseConnection.table("tiebaInfo")
NewsHTable = HBaseConnection.table("newsInfo2")
UserHTable = HBaseConnection.table("userInfo")


#2018-10:   625994
#2018-9,10: 1018778
#2018-11:395706
#2018-8-:1853752
#2018-1,7,8,9,10,11: 3338169
#2018-3,4,5: 4194303
#2018-:4799884
#2017-11,10:5957517
#2017-7,89:5957846
#2017-all
#2018-2:6533370
#2016-12:6533408
#2016-8.9.10.11:6533583
#2016-1~7:6533837
#2018-11:6689350(time out)
#2018-12:6715510(time out)
#2018-6:7248720
#2015-6~12:8083960
#2015-1-5:8084115
#2014-1-5:8084192
#2014-8-12:8084178
#2018-12:8182334
#2014-2018:10278078
event = 10278078
ed_dict  = {}#事件时间
eid_dict = {}#事件id
etype = {}#事件type
#months = ["2014-6-","2014-7"]
months = ["2018-10-","2018-9-","2018-8-","2018-7-"]
def init_query():
    es_search_options = {
        "query": {
            "bool": {}
        },
    }
    return es_search_options


def set_query_time(es_search_options, timezone, begin_time, end_time):
    query_options ={
        "range": {
            timezone: {
                "gt": begin_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "lt": end_time.strftime("%Y-%m-%dT%H:%M:%S")
            }
        }
    }
    if "must" not in es_search_options["query"]["bool"]:
        es_search_options["query"]["bool"]["must"] = []
    es_search_options["query"]["bool"]["must"].append(query_options)
    return es_search_options


def getWeiboText(id):
    res = requests.get("http://bd36:9200/crawler_all/msg/" + id)
    record = json.loads(res.content)
    text = ""
    try:
        source = record['_source']
        dateinfo = dateparser.parse(source['releasedate']) + datetime.timedelta(hours=8)
        row = WeiboHTable.row(dateinfo.strftime("%Y%m%d%H%M%S") + "-" + record['_id'])
        # print(row["info:text"])
        for key in row.keys():
            text = row["info:text"]
            #print(key+": " + row[key])
    except Exception as e:
        print("--------------------")
        print(e)
    return text

def getWeiboIDs(begin, end):
    ids = []
    es_search_options = init_query()
    es_search_options = set_query_time(es_search_options, "releasedate", begin, end)
    if begin < datetime.datetime(2018, 5, 16):
        index = "crawler_all"
    else:
        index = "crawler-" + begin.strftime("%Y-%m-%d")
    res = helpers.scan(
        client=ES,
        query=es_search_options,
        size=10000,
        scroll='5m',
        index=index,
        doc_type="msg"
    )
    for data in res:
        ids.append(data["_id"])
    return ids


def getWechatText(id):
    text = ""
    try:
        row = WechatHTable.row(id)
        text = row["info:text"]
    except Exception as e:
        print (e)
    return text


def getWechatIDs(begin, end):
    ids = []
    es_search_options = init_query()
    es_search_options = set_query_time(es_search_options, "time", begin, end)
    res = helpers.scan(
        client=ES,
        query=es_search_options,
        size=10000,
        scroll='5m',
        index="wechat_v3",
        doc_type="msg"
    )
    for data in res:
        ids.append(data["_id"])
    return ids


def getTiebaText(id):
    text = ""
    try:
        row = TiebaHTable.row(id)
        # print(row["info:title"])
        text = row["info:text"]
    except Exception as e:
        print (e)
    return text


def getTiebaIDs(begin, end):
    ids = []
    es_search_options = init_query()
    es_search_options = set_query_time(es_search_options, "time", begin, end)
    res = helpers.scan(
        client=ES,
        query=es_search_options,
        size=10000,
        scroll='5m',
        index="tieba_info2",
        doc_type="msg"
    )
    for data in res:
        ids.append(data["_id"])
    return ids


def getNewsText(id):
    text = ""
    try:
        row = NewsHTable.row(id)
        # text = json.dumps(row, ensure_ascii=False, indent=2)
        text = row['info:text']
    except Exception as e:
        print(e)
    return text

def getNewsIDs(begin, end):
    ids = []
    datas = []
    es_search_options = init_query()
    es_search_options = set_query_time(es_search_options, "time", begin, end)
    res = helpers.scan(
        client=ES,
        query=es_search_options,
        size=10000,
        scroll='5m',
        index="news_info3",
        doc_type="msg"
    )
    for data in res:
        ids.append(data["_id"])
        datas.append(data["_source"]["time"])
    return ids,datas

def getNews(begin,end):
    try:
        ids,datas = getNewsIDs(begin,end)
        l = len(ids)
    except:
        return
    for i in range(l):
        id = ids[i]
        d = datas[i]
        try:
            global event,ed_dict,eid_dict,etype
            row = NewsHTable.row(id)
        #    keywords = row['info:corewords']
            text = row['info:text'].replace("\n","")
            ed_dict[event] = d
            eid_dict[event] = id
            etype[event] = row['info:longtype2']
            with open("news/event" + str(event) + ".txt","w") as f:
         #       f.write(keywords+ "\n")
                f.write(text + "\n")
                event += 1

            print(event)
        except Exception as e:
            pass
'''
def getNews(begin,end):
    ids,datas = getNewsIDs(begin,end)
    l = len(ids)
    for i in range(l):
        id = ids[i]
        d = datas[i]
        
        global event,ed_dict,eid_dict,etype
        row = NewsHTable.row(id)
        keywords = row['info:corewords']
        text = row['info:text'].replace("\n","")
        ed_dict[event] = d
        eid_dict[event] = id
        etype[event] = row['info:longtype2']
        with open("news/event" + str(event) + ".txt","w") as f:
            f.write(keywords+ "\n")
            f.write(text + "\n")
            event += 1

        print(event)
'''
if __name__ == '__main__':
    
    #ids = getNewsIDs(dateparser.parse("2018-11-01 00:00:00"), dateparser.parse("2018-11-01 00:01:00"))

    '''
    print(ids[0])
    for key in NewsHTable.row(ids[0]).keys():
		print(key+": "+NewsHTable.row(ids[0])[key])

    tieba_ids = getTiebaIDs(dateparser.parse("2018-11-01 00:00:00"), dateparser.parse("2018-11-01 00:01:00"))
    for key in TiebaHTable.row(tieba_ids[0]).keys():
        print(key+": "+TiebaHTable.row(tieba_ids[0])[key])
    weibo_ids = getWeiboIDs(dateparser.parse("2018-11-01 00:00:00"), dateparser.parse("2018-11-01 00:01:00"))
    for key in WeiboHTable.row(weibo_ids[0]).keys():
        print(key+": "+WeiboHTable.row(weibo_ids[0])[key])

#    print(getWeiboIDs(dateparser.parse("2017-11-01 00:00:00"), dateparser.parse("2017-11-01 00:01:00"))[0])
#    getWeiboText(getWeiboIDs(dateparser.parse("2018-11-01 00:00:00"), dateparser.parse("2018-11-01 00:01:00"))[0])
     
    p = Pool(30)
    result = []
    for i in range(30):
        begin = dateparser.parse(month+str(i+1)+" 00:00:00")
        end = dateparser.parse(month+str(i+1)+" 23:59:59")
        print("process{0} start".format(i))
        result.append(p.apply_async(getNews,args=
            (begin,end)))
    p.close()
    p.join()
    for r in result:
        print(r.get())
    print("Done!")
     '''
    for month in months:
       for i in range(30):
           begin = dateparser.parse(month+str(i+1)+" 00:00:00")
           end = dateparser.parse(month+str(i+1)+" 23:59:59")
           getNews(begin,end)

       with open("dict/ed_dict_"+month+".txt","w") as f:
           f.write(str(ed_dict))

       with open("dict/eid_dict_"+month+".txt","w") as f:
           f.write(str(eid_dict))

       with open("dict/etype_dict_"+month+".txt","w") as f:
           f.write(str(etype))





'''
info:corewords : 莱斯 南方哈佛大学 大学 美国 可以免除 学费 家庭 科学 世界 教育 申请 工程 计划 有着 可以
info:url : http://www.sohu.com/a/257164173_115801#comment_area
info:join : 0
info:post_user : 美嘉留学陈华
info:uid : 8
info:releasedate : 20181001080001
info:reply : 0
info:imgurl : http://sucimg.itc.cn/avatarimg/9c6474a28fb349919f339760dfd88ea2_1440464854196 
http://5b0988e595225.cdn.sohucs.com/images/20180930/5d42d74901ce47c5bc71c2126a72058f.jpeg 
http://5b0988e595225.cdn.sohucs.com/images/20180930/5edb3294c9494acc82dc5821aa488a87.jpeg 
http://5b0988e595225.cdn.sohucs.com/images/20180930/46a0011c3fa548cb97ac29da8c234847.jpeg 
http://5b0988e595225.cdn.sohucs.com/images/20180930/4beb5ef1b607409abd8df57d64baa493.png 
http://sucimg.itc.cn/avatarimg/9c6474a28fb349919f339760dfd88ea2_1440464854196 
http://29e5534ea20a8.cdn.sohucs.com/c_fill,w_220,h_110,g_faces/c_cut,x_36,y_0,w_607,h_404/os/news/773c98e0f24d533a5d2df9b94f80bb5d.jpg 
http://29e5534ea20a8.cdn.sohucs.com/c_fill,w_220,h_110,g_faces/c_cut,x_105,y_0,w_1377,h_918/os/news/ff5a129f0ae7fa5c1bf6bbc94df2a959.jpg 
http://29e5534ea20a8.cdn.sohucs.com/c_fill,w_220,h_110,g_faces/c_cut,x_0,y_0,w_751,h_500/os/news/87f24dd252431cb9c688e6b0bc36d1bd.jpg 
http://29e5534ea20a8.cdn.sohucs.com/c_fill,w_220,h_110,g_faces/c_cut,x_0,y_0,w_513,h_342/os/news/12c955a75b2430a383662447c25f6c34.jpg 

info:title : 好消息！美国“南方哈佛大学”为家庭收入低的学生免学费！ 
info:longtype2 : 60
info:text : 原标题：好消息！美国“南方哈佛大学”为家庭收入低的学生免学费！
美国一直是中国留学生的首选留学国，但是居高不下的留学费用，一直是很多家庭的难点和重点。最近，有着“南方哈佛大学”美誉的莱斯大学于9月18日宣布了一项重大消息，只要家庭年收入在20万美元以下，就可以享受多种优惠。

莱斯大学宣布，将从明年秋季开始实施这项被称为“莱斯投资”（Rice Investment）的计划，其目的是为了让那些有天赋的学生享有接受优质私立教育的机会，而不必被学费压垮了肩膀。这项计划的内容包括，对于家庭年收入在6万5千美元以下的学生，可以免除全部的学费、住宿费和各项杂费；对于家庭年收入在13万美元以下的学生，可以免除全部学费；而来自年收入在13万至20万美元之间的中产阶级家庭的孩子，也可以免除其一半的学费。
这项计划不仅对明年入学的新生适用，也适用于所有还未毕业的本科生。

莱斯大学（Rice University），为美国南方最高学府，美国大学协会（AAU）成员，是一所世界著名的私立研究型大学，”新常春藤“名校之一。在2019年USNews《美国新闻与世界报道》大学排名中，莱斯大学与康奈尔大学并列全美第16名。因其高质量的教育和不断取得的国际学术成就,而与斯坦福大学，加州理工学院，麻省理工学院等25所高校被称为“新常春藤”院校，受到越来越多的学生的青睐。
2017年USNews美国综合排名第15名，由11个学院和8个学术研究学院组成，拥有极高的科研水平，累计科研经费达1.153亿美元，以工程、管理、科学、艺术、人类学闻名，尤其以工程系最为优秀。

学校小而精致，本科学部以精英教育而著称，同时有着优秀的应用科学和工程学的研究生项目。莱斯大学在材料科学、纳米科学、人工心脏研究、以及太空科学领域占有世界领先的地位。
录取要求
本科：
GPA：3.5+
ACT：29-33
SAT：1300-1550
雅思：7.0
托福：100
硕士：
GPA：3.5+
GRE：320+
GMAT：680+
雅思：7.0
托福：100
申请日期：
秋季申请截止日：1月1日
提前决定申请截止日：11月1日
转学申请截止日：3月15日
'''
