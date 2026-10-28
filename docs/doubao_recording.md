<span id="602731ab"></span>
# 流程简介
妙计录音文件识别服务的处理流程分为提交任务和查询结果两个阶段
任务提交：提交音频链接并获取服务端返回的任务ID
结果查询：通过任务 ID 查询转写结果 [建议>30S轮询频次，产出时间取决音频大小和系统排队负载]
![Image](https://p9-arcosite.byteimg.com/tos-cn-i-goo7wpa0wc/80365e5725124cbf9e61403e33b00a96~tplv-goo7wpa0wc-image.image =185x)

<span id="1316b087"></span>
# 提交任务
<span id="fdc79cf6"></span>
## 接口地址
火山地址：https://openspeech.bytedance.com/api/v3/auc/lark/submit

<span id="f076cf8d"></span>
## 请求
请求方式：HTTP POST
超时时间：5秒
请求和应答，均采用在 HTTP BODY 里面传输 JSON 格式字串的方式。
Header 需要加入内容类型标识：

| | | | \
|Key |说明 |Value 示例 |
|---|---|---|
| | | | \
|X-Api-App-Key |使用火山引擎控制台获取的APP ID，可参考 [控制台使用FAQ-Q1](https://www.volcengine.com/docs/6561/196768#q1%EF%BC%9A%E5%93%AA%E9%87%8C%E5%8F%AF%E4%BB%A5%E8%8E%B7%E5%8F%96%E5%88%B0%E4%BB%A5%E4%B8%8B%E5%8F%82%E6%95%B0appid%EF%BC%8Ccluster%EF%BC%8Ctoken%EF%BC%8Cauthorization-type%EF%BC%8Csecret-key-%EF%BC%9F) |123456789 |
| | | | \
|X-Api-Access-Key |使用火山引擎控制台获取的Access Token，可参考 [控制台使用FAQ-Q1](https://www.volcengine.com/docs/6561/196768#q1%EF%BC%9A%E5%93%AA%E9%87%8C%E5%8F%AF%E4%BB%A5%E8%8E%B7%E5%8F%96%E5%88%B0%E4%BB%A5%E4%B8%8B%E5%8F%82%E6%95%B0appid%EF%BC%8Ccluster%EF%BC%8Ctoken%EF%BC%8Cauthorization-type%EF%BC%8Csecret-key-%EF%BC%9F) |your-access-key |
| | | | \
|X-Api-Resource-Id |表示调用服务的资源信息 ID，固定值volc.lark.minutes |volc.lark.minutes |
| | | | \
|X-Api-Request-Id |用于提交和查询任务的任务ID，推荐传入随机生成的UUID |67ee89ba-7050-4c04-a3d7-ac61a63499b3 |
| | | | \
|X-Api-Sequence |发包序号，固定值，-1 | |

<span id="7c7fb860"></span>
### 请求字段
注意必选属性和可选属性，附加的可选属性必须至少选择一个
1 语音转写
AudioTranscriptionEnable必须设置为true，即默认基础功能
 辅助参数SpeakerIdentification、NumberOfSpeaker按说明和实际情况赋值
2 附加功能[翻译、待办提取、流程句提取、问答提取、全文总结、章节总结]
需至少选择一个使用，否则提交会失败
3 附加功能开启说明：
翻译：TranslationEnable=true，需设置TargetLang，当前支持中英互转
待办提取：InformationExtractionEnabled=true，InformationExtractionParams.Types需包含todo_list
问答提取：InformationExtractionEnabled=true，InformationExtractionParams.Types需包含question_answer
全文总结：SummarizationEnabled=true，SummarizationParams.Types需包含summary
章节总结：ChapterEnabled=true
4 AllActivate 是否打包计费
 注意根据收费模式赋值，若选项为true则按打包价收费，若为false则按选择的具体附加功能汇总计费

| | |||| \
|Method |POST | | | |
|---|---|---|---|---|
| | | | || \
| |Content-Type | |application/json | |
| | | | | | \
|Body |Input |必选 |object |输入参数 |
| | | | | | \
| |Input.Offline |必选 |object |离线参数 |
| | | | | | \
| |Input.Offline.FileURL |必选 |\
| | | |string |文件url  |\
| | | | |文件大小< 1G 时长2小时 |
| | | | | | \
| |Input.Offline.FileType |必选 |enum |文件类型  |\
| | | | |音频传递 audio |\
| | | | |视频传递 video |\
| | | | |视频格式支持：MP4、AVI、MKV、MOV、FLV、WMV |\
| | | | |音频格式支持：MP3、WAV、AAC、FLAC、OGG |
| | | | | | \
| |Params |必选 |object |音视频分析参数 |
| | | | | | \
| |Params.AllActivate |必选 |bool |是否打包计费 |\
| | | | |[非全功能使用，具体功能需设置设对应功能属性为true] |
| | | | | | \
| |Params.SourceLang |\
| | |必选 |\
| | | |enum |原始语种 |\
| | | | |zh_cn：中   |\
| | | | |en_us：英 |
| | | | | | \
| |Params.AudioTranscriptionEnable |必选 |bool |是否开启语音转写 默认写死true |
| | | | | | \
| |Params.AudioTranscriptionParams |必选 |object |语音转写参数 |
| | | | | | \
| |Params.AudioTranscriptionParams.SpeakerIdentification |必选 |bool |是否开启说话人识别  |
| | | | | | \
| |Params.AudioTranscriptionParams.NumberOfSpeaker |必选 |\
| | | |int |说话人数量，为0时算法自动识别 |\
| | | | |如果知道会议几个说话人可写，如果不知道默认写0 |
| | | | | | \
| |Params.AudioTranscriptionParams.HotWords |可选 |string |热词，数据格式："[{\"word\":\"热词\"}]" |
| | | | | | \
| |Params.AudioTranscriptionParams.NeedWordTimeSeries |必选 |bool |是否需要单词时间序列 |
| | | | | | \
| |Params.TranslationEnable |可选 [翻译]开启 |bool |是否翻译转写文本 |
| | | | | | \
| |Params.TranslationParams |[翻译]附加参数 |object |翻译参数 |
| | | | | | \
| |Params.TranslationParams.TargetLang |\
| | |[翻译]附加参数 |\
| | | |enum |\
| | | | |目标语言 |\
| | | | |zh_cn：中 |\
| | | | |en_us：英 |
| | | | | | \
| |Params.InformationExtractionEnabled |可选 [提取]开启 |bool |是否需要文章结构化数据 |
| | | | | | \
| |Params.InformationExtractionParams |[提取]附加参数 |object |文章结构化参数 |
| | | | | | \
| |Params.InformationExtractionParams.Types |\
| | |[提取]附加参数 |List enum |\
| | | | |todo_list : 待办提取 |\
| | | | |question_answer:问答提取 |
| | | | | | \
| |Params.SummarizationEnabled |可选[全文总结] 开启 |bool |是否开启全文总结 |
| | | | | | \
| |Params.SummarizationParams |[全文总结]附加参数 |object |全文总结参数 |
| | | | | | \
| |Params.SummarizationParams.Types |[全文总结]附加参数 |List enum |summary：全文总结 |
| | | | | | \
| |Params.ChapterEnabled |可选[章节总结] 开启 |bool |是否开启章节总结 |
| | | | | | \
| | |\
| | | |\
| | | | | |

<span id="5329bb04"></span>
### 示例
```JSON
{
    "Input": {
        "Offline": {
            "FileURL": "https://tosv.byted.org/obj/soundtools/TaskType_media_test_cases/vc.wav",
            "FileType": "audio"
        }
    },
    "Params": {
        "AllActivate":true,
        "SourceLang": "zh_cn",
        "AudioTranscriptionEnable": true,
        "AudioTranscriptionParams": {
            "SpeakerIdentification": true,
            "NumberOfSpeaker": 0,
            "HotWords": ""
        },
        "TranslationEnable": false,
         "TranslationParams": {
            "TargetLang": "zh_cn"
        },
        "InformationExtractionEnabled": true,
        "InformationExtractionParams": {
            "Types": [
                "todo_list",
                "question_answer",
            ]
        },
        "SummarizationEnabled": true,
        "SummarizationParams": {
            "Types": [
                "summary"
            ]
        },
        "ChapterEnabled": true
    }
}
```

<span id="a5c781e8"></span>
## 应答
Response header如下：

| | | | \
|Key |说明 |Value 示例 |
|---|---|---|
| | | | \
|X-Tt-Logid |服务端返回的 logid，建议用户获取和打印方便定位问题 |202407261553070FACFE6D19421815D605 |
| | | | \
|X-Api-Status-Code |提交任务后服务端返回的状态码，20000000表示提交成功，其他表示失败 | |
| | | | \
|X-Api-Message |提交任务后服务端返回的信息，OK表示成功，其他表示失败 | |

<span id="a7af2b71"></span>
### 响应字段

| | | | \
|字段 |类型 |说明 |
|---|---|---|
| | | | \
|Data |object |结果 |
| | | | \
|Data.TaskID |string |任务ID |

<span id="f8c84d98"></span>
### 示例
```JSON
{
    "Data": {
        "TaskID": "7534288318142352914"
    }
}
```

<span id="e9f94b3e"></span>
# 查询结果
<span id="d3608e15"></span>
## 接口地址
火山地址：https://openspeech.bytedance.com/api/v3/auc/lark/query
<span id="7e97f106"></span>
## 请求
请求方式：HTTP POST
超时时间：5秒
请求和应答，均采用在 HTTP BODY 里面传输 JSON 格式字串的方式。
Header 需要加入内容类型标识：

| | | | \
|Key |说明 |Value 示例 |
|---|---|---|
| | | | \
|X-Api-App-Key |使用火山引擎控制台获取的APP ID，可参考 [控制台使用FAQ-Q1](https://www.volcengine.com/docs/6561/196768#q1%EF%BC%9A%E5%93%AA%E9%87%8C%E5%8F%AF%E4%BB%A5%E8%8E%B7%E5%8F%96%E5%88%B0%E4%BB%A5%E4%B8%8B%E5%8F%82%E6%95%B0appid%EF%BC%8Ccluster%EF%BC%8Ctoken%EF%BC%8Cauthorization-type%EF%BC%8Csecret-key-%EF%BC%9F) |123456789 |
| | | | \
|X-Api-Access-Key |使用火山引擎控制台获取的Access Token，可参考 [控制台使用FAQ-Q1](https://www.volcengine.com/docs/6561/196768#q1%EF%BC%9A%E5%93%AA%E9%87%8C%E5%8F%AF%E4%BB%A5%E8%8E%B7%E5%8F%96%E5%88%B0%E4%BB%A5%E4%B8%8B%E5%8F%82%E6%95%B0appid%EF%BC%8Ccluster%EF%BC%8Ctoken%EF%BC%8Cauthorization-type%EF%BC%8Csecret-key-%EF%BC%9F) |your-access-key |
| | | | \
|X-Api-Resource-Id |表示调用服务的资源信息 ID，固定值volc.lark.minutes |volc.lark.minutes |
| | | | \
|X-Api-Request-Id |用于提交和查询任务的任务ID。查询时需使用提交成功的任务Id |67ee89ba-7050-4c04-a3d7-ac61a63499b3 |

<span id="88347fad"></span>
### 请求字段

| | ||| \
|Method |POST | | |
|---|---|---|---|
| | | || \
| |Content-Type |application/json | |
| | | | | \
|Body |TaskID |string |任务ID |

<span id="706710e2"></span>
### 示例
```Plain Text
{"TaskID":"7534267175524109842"}
```

<span id="cebdc193"></span>
## 应答
Response header如下：

| | | | \
|Key |说明 |Value 示例 |
|---|---|---|
| | | | \
|X-Tt-Logid |服务端返回的 logid，建议用户获取和打印方便定位问题 |202407261553070FACFE6D19421815D605 |
| | | | \
|X-Api-Status-Code |提交任务后服务端返回的状态码，具体错误码参考下面错误码列表 | |
| | | | \
|X-Api-Message |提交任务后服务端返回的信息，OK表示成功，其他表示失败 | |

<span id="af2f6502"></span>
### 响应字段

| | | | \
|字段 |类型 |说明 |
|---|---|---|
| | | | \
|Code |int |状态码 |\
| | |0: 成功 |
| | | | \
|Message |string |状态 |
| | | | \
|Data |object |响应体 |
| | | | \
|Data.TaskID |string |任务ID |
| | | | \
|Data.ErrCode |int |任务错误码 |
| | | | \
|Data.ErrMessage |string |任务错误信息 |
| | | | \
|Data.Status |\
| |enum |任务状态 |\
| | |running：运行中 |\
| | |success：成功 |\
| | |failed：失败 |
| | | | \
|Data.Result |object |结果 |
| | | | \
|Data.Result.AudioTranscriptionFile |string |语音转写结果url 有效时间24小时 |
| | | | \
|Data.Result.ChapterFile |string |章节总结结果url 有效时间24小时 |
| | | | \
|Data.Result.InformationExtractionFile |string |文章结构化结果url 有效时间24小时 |
| | | | \
|Data.Result.SummarizationFile |string |全文总结结果url 有效时间24小时 |
| | | | \
|Data.Result.TranslationFile |string |翻译文本结果url 有效时间24小时 |

<span id="bf7ae3c3"></span>
### 示例
注意：Status为running说明任务还在计算中，当为success则会看到具体附加功能返回的文件URL[一小时有效]
另外任务本身超过24小时未结束则自动丢弃
```JSON
// 请求
{
    "TaskID": "7424817657534087169"
}
// 响应
{
    "Data": {
        "Result": {
            "AudioTranscriptionFile": "https://larkai-workflow-boe.tos-cn-beijing.volces.com/7535398005722058258.json?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTMjI4MTU3OTk1MTNjNDE5ZGI0ZmQyZGIwYjYxZDBkMDE%2F20250806%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Date=20250806T091247Z&X-Tos-Expires=86400&X-Tos-Signature=9f2d394e760265051ba37ac3d8e64e503cec9ca9b9de53ac492a0f13b132146b&X-Tos-SignedHeaders=host",
            "ChapterFile": "https://larkai-workflow-boe.tos-cn-beijing.volces.com/7535398006363786770.json?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTMjI4MTU3OTk1MTNjNDE5ZGI0ZmQyZGIwYjYxZDBkMDE%2F20250806%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Date=20250806T091248Z&X-Tos-Expires=86400&X-Tos-Signature=6fa3052dd6d0b822017f22d16ba439d795926424a0902d33ad82d924d94d9d3a&X-Tos-SignedHeaders=host",
            "InformationExtractionFile": "https://larkai-workflow-boe.tos-cn-beijing.volces.com/7535398006493810194.json?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTMjI4MTU3OTk1MTNjNDE5ZGI0ZmQyZGIwYjYxZDBkMDE%2F20250806%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Date=20250806T091248Z&X-Tos-Expires=86400&X-Tos-Signature=53df4799d0c3a66e094f431bd916befac86aefc223b693019da2a96dd5eef63f&X-Tos-SignedHeaders=host",
            "SummarizationFile": "https://larkai-workflow-boe.tos-cn-beijing.volces.com/7535398006112128530.json?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTMjI4MTU3OTk1MTNjNDE5ZGI0ZmQyZGIwYjYxZDBkMDE%2F20250806%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Date=20250806T091248Z&X-Tos-Expires=86400&X-Tos-Signature=f0565568ffc34f3fd6489593e8e8b69e98ad47a3129c66e59620e2550399ad9e&X-Tos-SignedHeaders=host",
            "TranslationFile": "https://larkai-workflow-boe.tos-cn-beijing.volces.com/7535398005914996242.json?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTMjI4MTU3OTk1MTNjNDE5ZGI0ZmQyZGIwYjYxZDBkMDE%2F20250806%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Date=20250806T091247Z&X-Tos-Expires=86400&X-Tos-Signature=8cf7f8f117c377d554d10c64912f1cbf49a30358296e5e8a850554d9f52a6521&X-Tos-SignedHeaders=host"
        },
        "Status": "success",
        "TaskID": "7535397168945827368"，
        "ErrCode": 0,
        "ErrMessage": ""
    }
}
```

<span id="4ff65de1"></span>
### 任务错误码

| | | \
|ErrCode |说明 |
|---|---|
| | | \
|4004 |文件大小超限 |
| | | \
|4801 |空音频 |
| | | \
|4802 |文本格式错误 |
| | | \
|4803 |不支持的语种 |
| | | \
|4804 |空文本 |
| | | \
|4805 |空文件 |
| | | \
|4806 |没有可用时长 |
| | | \
|4807 |音频长度超限 |
| | | \
|4808 |不支持的音频格式 |
| | | \
|4809 |url无效 |
| | | \
|4810 |下载超时 |
| | | \
|4811 |下载错误 |
| | | \
|4812 |文件大小超限 |
| | | \
|4813 |不支持的语种 |


<span id="eae00449"></span>
### 语音转写 AudioTranscriptionFile
```JSON
数据结构:
[
    {
        "sentence_id":"1",       // 文本ID,有序的
        "paragraph_id":"1",      // 分段ID
        "speaker": {
            "id": 1,              // 说话人ID
            "name": "",           // 说话者名称
            "type": "" ,          // 说话者类型
        }
        "content":"",            // 内容
        "lang":"",               // 语言
        "start_time":"",         // 开始时间 第ms
        "end_time":"",          // 结束时间 第ms
    }
]
```

```JSON
[{"sentence_id":"0","paragraph_id":"","lang":"zh_cn","content":"是我们的人来了吗？","start_time":2250,"end_time":3330,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"1","paragraph_id":"","lang":"zh_cn","content":"任性。","start_time":27910,"end_time":28190,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"2","paragraph_id":"","lang":"zh_cn","content":"行，那我们就自己先过，先过一遍，后面再跟他们同步，然后先是那个这部分没什么更新。","start_time":119330,"end_time":131170,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"3","paragraph_id":"","lang":"zh_cn","content":"没什么更新，是吧？","start_time":131770,"end_time":132690,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"4","paragraph_id":"","lang":"zh_cn","content":"OK，那主要过一下这个监控大盘，我给大家聊一下这个监控文档。","start_time":133330,"end_time":138760,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"5","paragraph_id":"","lang":"zh_cn","content":"然后这个就是我们现在纪要的一个监控大盘，因为各个服务它都有自己的那个监控跟那个 SOA，所以这边关注的更多的是一个整体的链路，还有服务间的调动的关系。","start_time":145680,"end_time":160950,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"6","paragraph_id":"","lang":"zh_cn","content":"这上面的一监控，然后总体分为这么四个部分，第一个是总体监控，然后五个部分，第二个是纪要相关的，然后第三个是 Todo 相关的，第四个是章节相关的，最后一个是 segment 任务相关的，然后一个一个看一下总体监控，上面这些其实都是我们业务侧之前的那些打点，就是包括任 AI 任务提交、AI 任务数量堆积，然后包括 quota，还有这个 AI 任务的等待时间，这个点确定它为啥会，有时候会出现负值。","start_time":161510,"end_time":194940,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"7","paragraph_id":"","lang":"zh_cn","content":"诶，对了对了，茂哥，那个 AI 任务堆积和等待时间就能不能设一个那种标准线？","start_time":196540,"end_time":203940,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"8","paragraph_id":"","lang":"zh_cn","content":"就是比如说超过多少，就我们认为异常，这样可以就比较醒目一些，可以这个后面配置告警的时候就可以去做这件事。","start_time":204300,"end_time":217520,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"9","paragraph_id":"","lang":"zh_cn","content":"现在几个应该是有告警的，可以把那几个告警的值直接加起来就行。","start_time":217560,"end_time":221640,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"10","paragraph_id":"","lang":"zh_cn","content":"然后下面是第一个这个 Tab 是从业务侧的视角来看 AI 任务的失败，然后包括它提交失败，包括这个 error code 返回值呃。","start_time":223390,"end_time":238790,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"11","paragraph_id":"","lang":"zh_cn","content":"所以看到今天下午还是有蛮多130921309113010，还有非102这个类型，几个类型的错误就是从业务侧视角来看是这样，然后再往右边这边是从业务侧的视角来看它的生成时间，这个生成时间现在准吗？","start_time":238950,"end_time":255550,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"12","paragraph_id":"","lang":"zh_cn","content":"我看小时了。","start_time":256860,"end_time":258540,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"13","paragraph_id":"","lang":"zh_cn","content":"对啊，我看这个怎么好长时间是不取得。","start_time":258980,"end_time":261780,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"14","paragraph_id":"","lang":"zh_cn","content":"那个 sum 应该是 a 为 average 用 average 是吧？","start_time":261860,"end_time":265740,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"15","paragraph_id":"","lang":"zh_cn","content":"试一下有这么快吗？","start_time":265780,"end_time":271450,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"16","paragraph_id":"","lang":"zh_cn","content":"差不多现在那个 OK 了，summary 的话大概就20秒左右，然后 OK，那大概一分钟可以，对，那差不多他正在一分钟，然后 summary 大概20秒，对，那就对呃。","start_time":272090,"end_time":285130,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"17","paragraph_id":"","lang":"zh_cn","content":"然后这边这个是从达芬奇 pipeline 的视角看到的这个任务运行的数量，这个包括这个 meeting summary after meeting，还有这个 meeting agenda。","start_time":285970,"end_time":297530,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"18","paragraph_id":"","lang":"zh_cn","content":"诶，","start_time":297890,"end_time":298930,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"19","paragraph_id":"","lang":"zh_cn","content":"这个图是不？","start_time":298930,"end_time":299650,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"20","paragraph_id":"","lang":"zh_cn","content":"我们只保留我们那两个就行，就 meeting agenda 和 meeting summary 这个对，那就 pipeline，因为它里面包括其他的对，也包括其他这个没关系，这个我先放在这，嗯，然后这个再右边是从达芬奇 pipeline 的视角，它的一个失败的一个 status code。","start_time":299650,"end_time":322410,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"21","paragraph_id":"","lang":"zh_cn","content":"然后这个话比我们业务侧的视角可能要更加的就稍微再细一点，就是它会告诉你这是纪要生成的报错，还是 agenda 这个 pipeline 的报错。","start_time":322410,"end_time":334810,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"22","paragraph_id":"","lang":"zh_cn","content":"然后报错 code 那就放在这里呃。","start_time":335310,"end_time":338260,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"23","paragraph_id":"","lang":"zh_cn","content":"然后再往下这个就是纪要章节和 Todo，然后这三个监控的话，基本上照的原则就是最上一层是业务侧的监控，然后中间一层是达芬奇侧看到监控。","start_time":338260,"end_time":350700,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"24","paragraph_id":"","lang":"zh_cn","content":"最下面一层是从 INFO call 看到的一些线索，但 INFO call 现在只有 Pre 的点，所以这个 Pre INFO call 的点还比较少，要到周四的时候才能发到 online 去。","start_time":351020,"end_time":360060,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"25","paragraph_id":"","lang":"zh_cn","content":"然后业务侧监控跟上面其实差不多，只是把这个纪要相关的拎出来，然后这个包括纪要我从我们这边去提交任务的失败重试，然后从我们这边观察到这个耗时，这个应该也要改成 average，对吧？","start_time":360500,"end_time":378180,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"26","paragraph_id":"","lang":"zh_cn","content":"然后再下面这边就是这三列，这三行就是这三列，这三列就是从达芬奇视角来看一个纪要它生成，从达芬奇那边来看分为好多的 stage，包括这个 root、summary、transcribe，然后这里面其实有 Todo、basic、info 这种，这些 stage 的话，这边是一个提任务运行的一个数量。","start_time":383890,"end_time":409540,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"27","paragraph_id":"","lang":"zh_cn","content":"然后中间是各个 stage 的失败，然后面会打上这个 object ID，也就说这篇看上来这有几十篇，从就下午2:50开始到现在，其实一直有几十篇的这个纪要，它是有不停有报错的，然后再往这边就是，诶，茂哥，我这里有个疑问，就是这些 stage 就是纪要各 stage 左边那些哪些是强依赖？","start_time":409860,"end_time":437180,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"28","paragraph_id":"","lang":"zh_cn","content":"嗯，好问题这个其实我也不太确定，能看到的应该有一些是最主要的三个，就是这对于纪要来说就是 summary 是一个强依赖，对于 Todo 来说 Todo 这个就是一个强依赖。","start_time":438820,"end_time":455550,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"29","paragraph_id":"","lang":"zh_cn","content":"对啊对啊，他那个 trans transcript 的那个也是强依赖。","start_time":456350,"end_time":460350,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"30","paragraph_id":"","lang":"zh_cn","content":"这个后面可能我要整理一下。","start_time":461630,"end_time":463350,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"31","paragraph_id":"","lang":"zh_cn","content":"对于张杰的话，下面那个真的应该就是个强依赖。","start_time":464380,"end_time":466980,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"32","paragraph_id":"","lang":"zh_cn","content":"所以这些 stage 它报错的话，理论上都应该都是要重试的，其他的我不太确定，有可能有的是弱依赖，它直接就忽略过去了，对吧？","start_time":469740,"end_time":478540,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"33","paragraph_id":"","lang":"zh_cn","content":"但是对，有些弱依赖它那个就打一个报错就接，就继续执行。","start_time":479340,"end_time":484020,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"34","paragraph_id":"","lang":"zh_cn","content":"然后这个是各个 stage 的耗时，这个是不是应该也改成 s average？","start_time":485430,"end_time":489350,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"35","paragraph_id":"","lang":"zh_cn","content":"然后再往下是 info call 这边从 info call 的视角来看的一个那个镜头，然后这里面的在 info call 那边我们纪要的生成实际上就只有 summary GEN 这一个函数，然后 summary GEN 下面其实会调这个 LPP，就是 lambda Gateway，然后这里面带以 Lark AI 这个 tag 开头的这些埋点，其实都是在调用那个 Lambda Gateway，你可以看到这边有调中文的和英文的，当然现在只有 Pre 的点，后面会周四上完 online 之后会有 online 的点。","start_time":498120,"end_time":538330,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"36","paragraph_id":"","lang":"zh_cn","content":"然后这边就是 summary 这边的报错，然后这个错误码其实我加了，我看了一下这个错误码的意思是这个会太短了，然后生成的那个文字太少，所以他没有生成纪要。","start_time":538850,"end_time":555380,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"37","paragraph_id":"","lang":"zh_cn","content":"当然这个后续在我们这边其实他这个错误没往上面抛，这个错误其实在这可以在这里屏蔽掉，可以认为是一个正常现象。","start_time":555620,"end_time":563500,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"38","paragraph_id":"","lang":"zh_cn","content":"然后再往右边就是各个这个阶段的耗时，就 summary 这边的耗时，诶，但是我看现在这 只有 Pre 的，还有调用那个 LPP 的耗时，是不应该也是 average？","start_time":564670,"end_time":580380,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"39","paragraph_id":"","lang":"zh_cn","content":"然后在中间就是 Todo，Todo，嗯，没，只有达芬奇和那个 info call 这边因为 Todo 的提任务提交实际和 summer 在一块的。","start_time":587690,"end_time":599050,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"40","paragraph_id":"","lang":"zh_cn","content":"然后这边就是 Todo 的，它依赖两个 stage，一个是 Todo 和一个 Todo 键，然后 Todo 会有 stage 的失败错误码，还有后面 object ID。","start_time":599590,"end_time":609590,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"41","paragraph_id":"","lang":"zh_cn","content":"然后 Todo 各个这两个 stage 的这个耗时。","start_time":609590,"end_time":612510,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"42","paragraph_id":"","lang":"zh_cn","content":"诶，现在也挺快的，感觉他在 stage 这一侧感觉平均也就几秒钟，是吗？","start_time":617570,"end_time":624790,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"43","paragraph_id":"","lang":"zh_cn","content":"对，差不多 Todo，Todo 现在稍微会慢一点，我现在是大概两条。","start_time":625400,"end_time":630400,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"44","paragraph_id":"","lang":"zh_cn","content":"诶，那为什么从我们业务侧监控来看它要这么长时间？","start_time":633360,"end_time":637720,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"45","paragraph_id":"","lang":"zh_cn","content":"因为从单个纪要和这个这边，从我们看，从我们的视角来看它都要30多秒，但是从它 stage 的视角来看，就一个 stage 可能也就几秒钟，但是加起来可能就多了，因为他分了好多 state。","start_time":640460,"end_time":660790,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"46","paragraph_id":"","lang":"zh_cn","content":"那这个后面我们再看，后面我们再深入去看一下，然后再下面就是 info call 这边 Todo 的一个能力，Todo 一个监控，然后以 Lark AI 开头的这些其实都是在调用模型呃。","start_time":660790,"end_time":675510,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"47","paragraph_id":"","lang":"zh_cn","content":"然后剩下这两个是 info call 内部的一个，一个一个可以认为是一个函数，就是我这边包括这边耗时。","start_time":676250,"end_time":687340,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"48","paragraph_id":"","lang":"zh_cn","content":"OK，然后再下来就是章节这块，章节这块相对来说比较复杂，它依赖的这个 stage 特别的多，我们这边也是跟上刚才那个逻辑是一样，最上层是我们业务侧去提交这个章节任务，包括提交失败，然后包括我们章节，从我们的视角来看，这个章节任务的这个耗时，这个幸福方式你可以改成 P 90，可以试一下。","start_time":692940,"end_time":720950,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"49","paragraph_id":"","lang":"zh_cn","content":"因为最后可能就比较低了，选择 P90，就是刚才那个这个地方可以，P90，对，我看那个，对，P90可以试试，我们，对，这样看就对其他看到比较一致。","start_time":722220,"end_time":736380,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"50","paragraph_id":"","lang":"zh_cn","content":"那这个功能加一下，然后中间这块是从达芬奇视角来看章节的各个 stage，但它依赖的 stage 非常的多啊。","start_time":738720,"end_time":747520,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"51","paragraph_id":"","lang":"zh_cn","content":"然后这边我不知道为什么有一个 stage 它一直会报很多错，它是有这个绿色的，所以这边没把 object ID 打出来，如果打出来实在太多了，这个 stage 就是，诶，张瑞，你帮忙记个 Todo 吧。","start_time":749500,"end_time":765290,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"52","paragraph_id":"","lang":"zh_cn","content":"张伟，听到了吗？","start_time":768030,"end_time":769630,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"53","paragraph_id":"","lang":"zh_cn","content":"这个 video 第二个，这个 stage 有很多，一直在报这个1013090 video prices。","start_time":770710,"end_time":778390,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"54","paragraph_id":"","lang":"zh_cn","content":"好，就是这边，就是那个分段的那个，嗯，还有分段那个，对，1309那个好像是那其他的地方的报错，它只是一些偶现的一些或者说持续性的有几个零星的几个报错。","start_time":778510,"end_time":793350,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"55","paragraph_id":"","lang":"zh_cn","content":"虽然这个 stage 它一直在报，有非常多一直在报3060，因为它这个有团队办法，就设置有，反正分段结果可能还没分段完成。","start_time":793740,"end_time":804020,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"56","paragraph_id":"","lang":"zh_cn","content":"对，我觉得这个后面可以再看一下，统一改的。","start_time":804790,"end_time":807670,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"57","paragraph_id":"","lang":"zh_cn","content":"对，我们应该说你改是可以聊底下可以，对，在现在的时候可以，你别报考，没必要转，就是。","start_time":807990,"end_time":814030,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"58","paragraph_id":"","lang":"zh_cn","content":"然后再下面就是，对，在整个章节的生成过程中去调 info call 一下，嗯，其实最主要的就是这个 additional gene，它这是嗯章节里面最重要的一个 stage。","start_time":814190,"end_time":830400,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"59","paragraph_id":"","lang":"zh_cn","content":"嗯，然后其他的可以我理解，可以都认为是一些弱依赖，包括什么 question host，question fatal，SIM，more process detect，然后一样的像这些在 Lark AI 开头的这些都。","start_time":830400,"end_time":845640,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"60","paragraph_id":"","lang":"zh_cn","content":"诶茂哥调模型这个真的 summary 应该就是去给张杰生成这个要和博瑞斯。","start_time":845680,"end_time":852400,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"61","paragraph_id":"","lang":"zh_cn","content":"诶茂哥，我其实有个建议就是，嗯，那个诶张瑞，你把你麦关一下。","start_time":852600,"end_time":857510,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"62","paragraph_id":"","lang":"zh_cn","content":"茂哥，我有个建议就是比如像我们不熟悉的时候，其实比如说很多 stage 你把它都放一起，我们可能不知道看啥，所以有没有可能你后面强依赖梳理出来了之后，把强依赖单独做一个面板？","start_time":859000,"end_time":874030,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"63","paragraph_id":"","lang":"zh_cn","content":"嗯，这 LLPP 你可以认为都是强依赖，只不过从我们各个层级有非常多的重试，就它自身调用这 LPP 失败他也自己也会去重试，所以你要单列出来。","start_time":875850,"end_time":890140,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"64","paragraph_id":"","lang":"zh_cn","content":"嗯，感觉有点乱，因为它就是整个，你看这个 stage 非常多，包括到了 info call 这边它也非常多，你要是把强弱依赖都摘出来，这个就看起来很乱，而且后续维护起来可能也比较困难。","start_time":891020,"end_time":909660,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"65","paragraph_id":"","lang":"zh_cn","content":"然后这边是章节的调研事件，嗯，然后这边有一个报错，就9003，这我看了一下，就是因为会议时间太短，然后没能生成章节。","start_time":911220,"end_time":921900,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"66","paragraph_id":"","lang":"zh_cn","content":"这个后面也可以认为是一种正常现象，从我们这边把这个也不去作为一个，就是异常显示在这里，然后这边就是各个在 info call 里面对于各个环节的这个处理的耗时，对，这是章节。","start_time":922260,"end_time":942860,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"67","paragraph_id":"","lang":"zh_cn","content":"然后下面就是 segment 任务，segment 任务，然后最左边这个是任务的提交，然后它运行成功了会有一个点。","start_time":943580,"end_time":953300,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"68","paragraph_id":"","lang":"zh_cn","content":"然后就是相当于是它正确地返回了，然后这边第二列是 segment 任务本身会有一些异常退出，这边会打出 object ID 和 job ID。","start_time":953989,"end_time":965970,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"69","paragraph_id":"","lang":"zh_cn","content":"然后再往右边，这个是 segment 任务总体的一个运行的数量，然后 segment 任务本身它也分为好多个阶段，这些阶段的一个处理的耗时 PCC90。","start_time":965970,"end_time":979840,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"70","paragraph_id":"","lang":"zh_cn","content":"然后 segment 的任务，其实它最强的一个依赖就是一个叫 REF，REF 点 video 这个它也是个 info call 的插件。","start_time":984360,"end_time":994120,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"71","paragraph_id":"","lang":"zh_cn","content":"然后在这个插件里面去加了一些埋点，然后这插调用插件时候它里面也有很多的步骤，然后把这个相关的这个点也打出来了。","start_time":994450,"end_time":1004810,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"72","paragraph_id":"","lang":"zh_cn","content":"然后 react queue，那就这个 video 这个插件里面本身的一些报错也在这里打出来，然后再后面就是调它的超时。","start_time":1005140,"end_time":1014620,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"73","paragraph_id":"","lang":"zh_cn","content":"需求啊。","start_time":1019310,"end_time":1023170,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"74","paragraph_id":"","lang":"zh_cn","content":"最后一列实际上是我们任务平台的监控，就是 segment 任务本身，它重试五次以后达到上限，最终放弃掉了。","start_time":1023650,"end_time":1033330,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"75","paragraph_id":"","lang":"zh_cn","content":"但其实可以看到每天还是有那么几个，大概四五个吧。","start_time":1034060,"end_time":1038980,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"76","paragraph_id":"","lang":"zh_cn","content":"我看每天，所以理论上这四五个任务，这四五个分段都是有问题的。","start_time":1038980,"end_time":1043540,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"77","paragraph_id":"","lang":"zh_cn","content":"他的这个 segment 任务重试了好几次，已经重试5次，已经全部退出来了，所以这个后续可能还要找 AI 人，再具体看一下是啥原因。","start_time":1044130,"end_time":1054450,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"78","paragraph_id":"","lang":"zh_cn","content":"理论上那几个它分段应该都是不成功的，我理解。","start_time":1055970,"end_time":1060170,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"79","paragraph_id":"","lang":"zh_cn","content":"然后这个是总体的一个监控大盘，然后大家看一下，如果有什么漏掉的，或者是需要补充的或者是建议的，也可以提一下。","start_time":1062790,"end_time":1073680,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"80","paragraph_id":"","lang":"zh_cn","content":"后续我们就是把相应的一些监控或者埋点给它加进来，然后再拉一下这个单个任务，单个任务其实和刚才的逻辑基本上是类似的呃。","start_time":1073680,"end_time":1091150,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"81","paragraph_id":"","lang":"zh_cn","content":"第一列这个我是把各个阶、各个环节的一些处理的失败都放在这里，在这个地方把 meeting ID 填进去之后，这个是我拉的下午的一个例子，把这个 meeting ID 填进去之后，他如果在某个环节上会有失败的话，然后在这一行，第一行里面都会有相应的埋点出来。","start_time":1093490,"end_time":1111170,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"82","paragraph_id":"","lang":"zh_cn","content":"比如说这个，嗯，从我们业务侧来看，就是它有一个报错是13010，然后在达芬奇这边它实际上是可以看到是这个纪要任务，然后它有报错13110，然后在 stage 这边就是其实是 Todo 这个 stage 报了这个报错，那是不是 INFO call 还不太确定，因为这个是 online 的意思。","start_time":1111490,"end_time":1133560,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"83","paragraph_id":"","lang":"zh_cn","content":"info call 的打点还没上到 online。","start_time":1133640,"end_time":1136560,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"84","paragraph_id":"","lang":"zh_cn","content":"嗯。","start_time":1136560,"end_time":1136840,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"85","paragraph_id":"","lang":"zh_cn","content":"然后这边再往下就是纪要的各个 stage 的埋点，就是它调用到了这些 stage 就会有点出来，然后如果调用失败也会有点出来，包括它的耗时。","start_time":1138500,"end_time":1151140,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"86","paragraph_id":"","lang":"zh_cn","content":"可以在看到在这篇妙记里面，它的 summary 耗时大概是11秒左右，Todo 大概是30秒左右生成，然后其他的一些 stage 的耗时也都打在这里，再往下就是 info code 现在还看不到。","start_time":1151180,"end_time":1168340,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"87","paragraph_id":"","lang":"zh_cn","content":"然后一个是 online，还有这个 Todo 的，诶？","start_time":1168380,"end_time":1171860,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"88","paragraph_id":"","lang":"zh_cn","content":"Todo 的这个 stage 怎么耗时没有啊？","start_time":1172020,"end_time":1173820,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"89","paragraph_id":"","lang":"zh_cn","content":"对，我下来再看一下，有些点好像因为没人查被屏蔽掉了。","start_time":1174810,"end_time":1180090,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"90","paragraph_id":"","lang":"zh_cn","content":"然后就张杰的各个 stage 它的调用，还有各个 stage 的耗时最长的这个叫 video。","start_time":1180490,"end_time":1188220,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"91","paragraph_id":"","lang":"zh_cn","content":"他说，诶，不是这个，就我真的他花了26秒8再往下就是这个任务的 segment，这个妙记的 segment 任务，然后他任务提交可以看到是在2点三十六分左右，然后到了过了十就几分钟它运行成功了，然后如果它有异常退出的话，我会在这里导出来。","start_time":1188220,"end_time":1214710,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"92","paragraph_id":"","lang":"zh_cn","content":"然最后就是 segment 的任务去调那个 in info count 插件相关的一些打点。","start_time":1215510,"end_time":1220470,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"93","paragraph_id":"","lang":"zh_cn","content":"然后大体上就是这些内容，大家有没有什么问题？","start_time":1225410,"end_time":1232560,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"94","paragraph_id":"","lang":"zh_cn","content":"info call 去调 LPP 的时候，如果有报错，就在刚才那个地方也能显示出来，对，在这个单篇的里面也能显示出来，就会在这个 info call 调用失败这一栏，这一列，这个栏，这，这个是根据那个什么错误码去盘，对，如果调用 LPP，它返回码不是0的话就会导出来。","start_time":1232680,"end_time":1250680,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"95","paragraph_id":"","lang":"zh_cn","content":"还是说那个强依赖？","start_time":1266790,"end_time":1267790,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"96","paragraph_id":"","lang":"zh_cn","content":"我感觉可以先，就是比如说那个 summary，它有几个那个 state 是，比如说强依赖的感觉可以看能不能加一些那个标记，也行，在那个展示上你加个标记。","start_time":1267870,"end_time":1280890,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"97","paragraph_id":"","lang":"zh_cn","content":"可以啊，那没问题。","start_time":1281510,"end_time":1282510,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"98","paragraph_id":"","lang":"zh_cn","content":"对啊，黄敏泽。","start_time":1282590,"end_time":1283510,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"99","paragraph_id":"","lang":"zh_cn","content":"不用，先不用把面板分出来，这面板里头，嗯。","start_time":1285280,"end_time":1288680,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"100","paragraph_id":"","lang":"zh_cn","content":"行，那我保存一下。","start_time":1301290,"end_time":1303090,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"101","paragraph_id":"","lang":"zh_cn","content":"那我们继续。","start_time":1313410,"end_time":1314370,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"102","paragraph_id":"","lang":"zh_cn","content":"行，然后 SDC 这边的监控在这边也不过了，然后718的标品的现在还是正在弄，后续一些高卷什么都会同步到同步进去压测，这个上周应该也暂时没有更新，然后发布会相关的。","start_time":1322160,"end_time":1337700,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"103","paragraph_id":"","lang":"zh_cn","content":"嗯，OK，相关的我们还要过一下吗？","start_time":1338540,"end_time":1341190,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"104","paragraph_id":"","lang":"zh_cn","content":"已经开始搞了，对反，就是我下面，我那个就是那个纪要生成的那个兜底，那个我稍微简单过一下，我这边大概方案有。","start_time":1345450,"end_time":1355450,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"105","paragraph_id":"","lang":"zh_cn","content":"就上次评审完之后，那个思琪提说，就是我们到时候尽可能是去做一些自动的兜底，就尽量避免手动的一些处理，所以目前是把这个兜底方案，然后又重新梳理一下，然后现在大概的思路就是在这个这个思维角度，首先就是针对可能会出现异常 case 的情况，大概就这么多，然后罗列了一下，嗯，然后目前就是针对，比如说总结和待办。","start_time":1363620,"end_time":1390420,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"106","paragraph_id":"","lang":"zh_cn","content":"然后可能比如说他发起失败的话，目前我就会直接去把这个任务标记为成功，这时候他这边是可以编辑的，然后那个就是他们去直接去编辑就行了。","start_time":1390740,"end_time":1400420,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"107","paragraph_id":"","lang":"zh_cn","content":"然后如果是他这个任务处理失败的，但是有 call back 过来，然后这种情况下我也会判去判断，比如说如果是这个发布会，妙记也会直接去把它标记成功。","start_time":1400420,"end_time":1410170,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"108","paragraph_id":"","lang":"zh_cn","content":"嗯，然后页面是变成可编辑态。","start_time":1410330,"end_time":1412130,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"109","paragraph_id":"","lang":"zh_cn","content":"那另外一种情况就是说他这个 AI 那边他任务处理是卡死的，然后他一直没有 callback 过来。","start_time":1412450,"end_time":1418330,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"110","paragraph_id":"","lang":"zh_cn","content":"嗯，那这种情况下是需要我们手动去做一下那个恢复的，然后这个恢复的话其实就是模拟一次那个 callback 请求，然后把这个处理标记为一个成功，然后也是回到一个可编辑的状态，然后就大概总结和待办是这么去处理的。","start_time":1418370,"end_time":1432560,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"111","paragraph_id":"","lang":"zh_cn","content":"然后张杰的话，是啊，就是大概也是这么几种情况，然后如果是发起失败的话，就是直接标记为成功，然后这个时候是考虑是说使用一个 TC 的一个兜底配置，就是这个可以提前，比如说根据他们一些，比如发布会之前的一些这种彩排，然后去获取一个大概的一个分段，然后使用这个分段去兜底，然后配置，然后这个时候它页面变成一个可编辑态，如果有一些那个内容它可以去调整，但是分段的话就是会就确定下来了。","start_time":1432720,"end_time":1461180,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"112","paragraph_id":"","lang":"zh_cn","content":"对，然后如果有 call back，但是它是失败的，同样的也是使用这个 TCC 的兜底。","start_time":1461340,"end_time":1467460,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"113","paragraph_id":"","lang":"zh_cn","content":"然后如果是卡死没有 callback，那跟上面一样，就是我们手动去模拟一次 callback 调用，然后具体的分段话也是使用那个 TC 配置的那个去作为一个兜底的那个分段。","start_time":1467760,"end_time":1478760,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"114","paragraph_id":"","lang":"zh_cn","content":"对，所以现在就是目前就只有这两种情况下，需要我们手动去介入一下。","start_time":1479130,"end_time":1484330,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"115","paragraph_id":"","lang":"zh_cn","content":"对，然后对于编辑的场景现在应该是主要是两种，一种就是说他去新增那个待办，新增待办目前是他没办法自动关联时间嘛。","start_time":1485090,"end_time":1494360,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"116","paragraph_id":"","lang":"zh_cn","content":"然后这个地方我想了一下，其实可以在那个他待办的内容里面，可以，比如说定义一些，比如说他去在前面前缀加一些这个时间点。","start_time":1494400,"end_time":1502920,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"117","paragraph_id":"","lang":"zh_cn","content":"我们服务端根据这个妙记，是不是一些发布会妙记，然后去解析它前置的一些时间点，然后把它自动关联到这个时间点上去？","start_time":1503640,"end_time":1510520,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"118","paragraph_id":"","lang":"zh_cn","content":"比如说像这个例子，比如说我说一个什么10分20秒，这样带一个，比如说几号一个分隔，然后面就是那个 todo 内容，它会自动关联到这个10分20秒，然后内容就是这个，就具体的那个新增 todo 的内容，然后这个时候也会把这个就它会有一个手动添加的一个状态，然后自动会把它给移除掉，当成一个自动的那个添加。","start_time":1510720,"end_time":1530560,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"119","paragraph_id":"","lang":"zh_cn","content":"这样的话其实就不需要我们到时候手动去介入了，只需要去定一个这样规则就行。","start_time":1531360,"end_time":1535480,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"120","paragraph_id":"","lang":"zh_cn","content":"然后总结和张杰他如果去编辑的话，目前会有一个编辑的状态，然后这个也是考虑，就是说使用一个 TC 配置，然后去判断如果是特殊的妙记就不会去写这个已编辑的状态。","start_time":1536190,"end_time":1549110,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"121","paragraph_id":"","lang":"zh_cn","content":"然后如果他编辑之后的话，会自动地把一些那个，比如说之前那些那种类似于分布式锁的那些东西去写一下，然后屏蔽后续的一些，比如说达芬奇或 AI core 的一些这种 call back 回来，然后导致那些内容会覆盖掉。","start_time":1549790,"end_time":1564280,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"122","paragraph_id":"","lang":"zh_cn","content":"目前来讲大概思路就这样，然后这样就是这个方案更新之后，就是手动去触发的大概就两种场景，就是说没有课件要回来的，其他的就基本上就可以自动去覆盖掉所有这些异常 case。","start_time":1564840,"end_time":1577810,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"123","paragraph_id":"","lang":"zh_cn","content":"嗯，对，目前是反正基于这个方案来开发的，但是昨天晚上那个，就是那个重保那个周会上提到就是目前可能还没有确定的一个点，就是说我们就是一个一个就是说他去把那个链接暴露出来的，到底是使用那个文档的，还是使用妙记的？","start_time":1578250,"end_time":1600210,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"124","paragraph_id":"","lang":"zh_cn","content":"嗯，这个 PM 那边还没有跟发布的同学确定下来，另外一个就是说那个展示的内容是使用会中的还是使用那个？","start_time":1600290,"end_time":1607670,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"125","paragraph_id":"","lang":"zh_cn","content":"就后来那个伟琪做那个会后的这个暂时也没有完全确认，他这个还没定。","start_time":1607950,"end_time":1612670,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"126","paragraph_id":"","lang":"zh_cn","content":"对，如果可能目前，诶江凡，这我有个问题，你是说就是替，是说发布会展示的？","start_time":1613070,"end_time":1623690,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"127","paragraph_id":"","lang":"zh_cn","content":"嗯，","start_time":1623810,"end_time":1624090,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"128","paragraph_id":"","lang":"zh_cn","content":"纪要没确定，还是说替换的纪要没确定你替换的是什么意思呃？","start_time":1624330,"end_time":1631930,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"129","paragraph_id":"","lang":"zh_cn","content":"替换就是说它如果效果不好，我们不是用就是新的去替换。","start_time":1632730,"end_time":1639730,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"130","paragraph_id":"","lang":"zh_cn","content":"然后刚没听清你说的是我们对外展示的，嗯，就是还没确定用哪一方，还是说替换的内容还没确定用哪是对外展示的，这个还没有完全定下来，就是他现在还不确定到底是使用这个妙记的这个去承接，还是使用那个文档去承接。","start_time":1640030,"end_time":1660290,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"131","paragraph_id":"","lang":"zh_cn","content":"然后我印象那个文档里的纪要是会中提供的文档里面它是两种情况，一种就是说如果它会中开了一个什么免录制的一个功能，那它是使用会中的那个去生成的，那如果它没有开免录制，那就会使用那个会后的。","start_time":1661528,"end_time":1680298,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"132","paragraph_id":"","lang":"zh_cn","content":"就它是两种场景。","start_time":1680478,"end_time":1682878,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"133","paragraph_id":"","lang":"zh_cn","content":"那这两个就是昨天林燕那边还没有完全确认下来，应该是今天或明天会把这个再确认下来，然后可能确认下来之后，我们最终才能知道说我们这个这些兜底方案会不会能够正常用到，目前先按这些先去推进吧。","start_time":1685518,"end_time":1701518,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"134","paragraph_id":"","lang":"zh_cn","content":"我这边大概就更多那些做这个其实可以后来整理成一个 SOP 就是，对对对，是的，文档就是遇见什么样的情况，我们做什么样的操作。","start_time":1703298,"end_time":1715018,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"135","paragraph_id":"","lang":"zh_cn","content":"对，是的，这个反正等后面开发完，然后确定没有问题的话，就是在发布会之前会把这个 SOP 整理下来。","start_time":1715698,"end_time":1722018,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"136","paragraph_id":"","lang":"zh_cn","content":"SOP，诶？","start_time":1723818,"end_time":1724418,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"137","paragraph_id":"","lang":"zh_cn","content":"扩容的事那个事情后来怎么说？","start_time":1724618,"end_time":1727388,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"138","paragraph_id":"","lang":"zh_cn","content":"怎么 AI 那边都已经扩了吗？","start_time":1727388,"end_time":1730308,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"139","paragraph_id":"","lang":"zh_cn","content":"还是说那个就是大模型那边那个资源，对吧？","start_time":1730308,"end_time":1734918,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"140","paragraph_id":"","lang":"zh_cn","content":"对对对，那个我看昨天天豪那边更新的，应该是说那边应该可以满足我们目前的那个申请的一个诉求，这个应该看起来是问题不大，行。","start_time":1734918,"end_time":1746638,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"141","paragraph_id":"","lang":"zh_cn","content":"然后还有就是刚才那个预案的话，就是目前是，就是纪要这边是暂时，我这边这个预案应该是有了，然后还有就是什么包括录制 STT 那些，可能对，需要来补一个类似的，比如说如果现在现有的可能也是把那个类似于一些 SOP 的那些流程去梳理一下，嗯，行，这个我来吧。对，弄出来 SD。","start_time":1755218,"end_time":1777548,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"142","paragraph_id":"","lang":"zh_cn","content":"然后时间的话，应该张慧芳也记个 Todo，就录制和 SDK 的，我们发布会重保的相关的一些事情，时间应该是确定下来，就是9月4号。","start_time":1783998,"end_time":1794478,"speaker":{"id":"1","name":"说话人1","type":101},"words":null},{"sentence_id":"143","paragraph_id":"","lang":"zh_cn","content":"那我们今天会先这样。谢谢。","start_time":1805628,"end_time":1811088,"speaker":{"id":"1","name":"说话人1","type":101},"words":null}]
```

<span id="7b23b5bf"></span>
### 章节总结 ChapterFile
```JSON
{
    "chapter_summary":[           // 章节总结
        {
            "start_time": 26000,  // 开始时间
            "end_time": 1956000,  // 结束时间
            "title": "",          // 章节标题
            "summary": "",        // 章节总结
        }
    ]
}
```

```JSON
{"chapter_summary": [{"start_time": 2000, "end_time": 119000, "title": "会议开场", "summary": "会议开场"}, {"start_time": 119000, "end_time": 196000, "title": "自行先过内容，聚焦监控大盘及文档介绍 ", "summary": "本章节主要讨论工作安排，先自行过一遍内容，后续再与他人同步。重点介绍监控大盘和监控文档，监控大盘关注整体链路及服务间调动关系，总体分为总体监控、纪要相关、Todo 相关、章节相关、segment 任务相关五部分，还提及总体监控包含业务侧打点信息，需确定 AI 任务等待时间有时出现负值的原因。"}, {"start_time": 196000, "end_time": 409000, "title": "AI任务监控指标及数据处理讨论", "summary": "本章节主要讨论了AI任务相关监控设置。提议为AI任务堆积和等待时间设标准线用于配置告警，可将现有告警值相加。从业务侧和达芬奇pipeline视角分析任务失败、生成时间、运行数量、失败状态码等情况，还提及纪要章节和Todo监控，分业务侧、达芬奇侧、INFO call三层，INFO call Pre点较少，周四发online。"}, {"start_time": 409000, "end_time": 485000, "title": "纪要各阶段报错及强依赖情况探讨", "summary": "本章节提到从下午2:50开始有几十篇纪要一直存在报错，各stage会标注object ID。有人提出疑问，询问各stage哪些是强依赖，得到的回复是，对于纪要而言，summary、Todo、transcript是强依赖，这些stage报错理论上要重试，其他不太确定，有些可能是弱依赖，报错后或直接忽略，或打个报错继续执行，后续需整理。"}, {"start_time": 485000, "end_time": 564000, "title": "各阶段耗时显示调整及 info call 纪要生成情况说明", "summary": "本章节讨论了各 stage 耗时或改成 s average。从 info call 视角，纪要生成仅用 summary GEN 函数，其会调用 LPP（即 lambda Gateway），有以 Lark AI 开头的埋点，目前只有 Pre 点，周四上线后会有 online 点。还提到 summary 报错，错误码显示文本过短未生成纪要，该错误可屏蔽，视为正常现象。"}, {"start_time": 564000, "end_time": 633000, "title": "各阶段耗时情况及Todo任务相关信息", "summary": "本章节主要讨论了各阶段耗时情况，右侧是summary相关耗时，目前只有Pre和调用LPP的耗时，不确定是否应为平均值。中间提及Todo，达芬奇和info call的Todo任务提交与summer相关。Todo依赖两个stage，有失败错误码和object ID，还提到了这两个stage的耗时，整体在stage一侧平均耗时几秒，Todo稍慢，约两秒。"}, {"start_time": 633000, "end_time": 859000, "title": "业务监控时长问题及章节功能相关讨论与待办事项", "summary": "本章节主要讨论业务监控耗时问题，后续会深入查看。提及 info call 的 Todo 能力及监控，Lark AI 开头的在调用模型。章节部分较复杂，依赖众多 stage，建议将耗时统计方式改为 P90。记录 video 第二个 stage 报错问题待处理，章节生成中 additional gene 是重要 stage，其他多为弱依赖。"}, {"start_time": 859000, "end_time": 1062000, "title": "系统依赖展示建议及各任务情况汇报", "summary": "本章节提出将强依赖单独做面板的建议，认为强弱依赖都摘出会使界面乱且维护难。提到章节调研事件中报错 9003 因会议时间短，可不算异常。介绍了 segment 任务各列信息，其最强依赖是 REF 点 video 插件，还提及插件报错、超时等情况，指出每天有四五个 segment 任务重试 5 次后放弃，需找 AI 人员查明原因。"}, {"start_time": 1062000, "end_time": 1225000, "title": "监控大盘及单个任务埋点与耗时情况介绍 ", "summary": "本章节介绍了总体监控大盘，邀请大家提出补充和建议后会添加相应监控或埋点。还展示单个任务，其逻辑与之前类似，包含各环节处理失败埋点，如不同业务侧报错情况；有纪要各stage埋点、耗时信息，部分stage耗时需再核查；还有任务segment相关信息，如提交时间、运行情况及调info count插件打点。"}, {"start_time": 1225000, "end_time": 1313000, "title": "内容介绍、报错显示及标记添加讨论与决策", "summary": "本章节主要讨论了几个事项，一是 info call 调用 LPP 若报错会在单篇里的 info call 调用失败栏显示，根据错误码排查，返回码非 0 就会导出；二是考虑在 summary 有强依赖的几个 state 展示上加标记；三是决定目前不用把面板分出来，最后说话人表示要保存一下。"}, {"start_time": 1313000, "end_time": 1811000, "title": "发布会纪要兜底方案汇报及待确认事项沟通", "summary": "本章节主要讨论了SDC监控、718标品情况及发布会相关事宜。重新梳理了纪要生成兜底方案，明确多种异常情况处理方式，编辑场景提出自动关联时间等思路。目前有两个未确定点，需今明确认。AI扩容应能满足诉求，纪要预案已有，需补充录制STT等SOP流程，发布会重保时间为9月4号。"}]}
```

<span id="3507cfba"></span>
### 结构化数据 InformationExtractionFile
```JSON
{
    "question":[//问句提取
        {
            "sentence_id":"1",     // 文本ID
            "label":2,           // 文本类型：2（普通问句），3（重要问句）
        }
    ],
    "todo_list":[//待办
        {
            "content": "",                //待办内容
            "sentence_id": ["1","2"],     //相关文本ID
            "executor": "Kerwin",         //执行者名称
            "start_time": 478730,
            "execution_time": "无"
        }
    ]
}
```

todo_list : 待办提取 
question_answer:问答提取
```JSON
{"todo_list":[{"content":"整理纪要各 stage 中的强依赖关系","execution_ddl":"无","execution_time":["无"],"executor":["无"],"polished_res":{"content":"**强依赖关系整理**：整理纪要各 stage 中的强依赖关系，已知 summary、Todo、transcript 为强依赖，需进一步明确其他强依赖情况","execution_ddl":"无","executor":["无"]},"sentence_id":["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36"],"start_time":461630,"todo_idx":30},{"content":"深入分析 Todo 任务在业务侧监控中耗时较长的原因","execution_ddl":"无","execution_time":["无"],"executor":["无"],"polished_res":{"content":"**Todo 耗时分析**：深入分析 Todo 任务在业务侧监控中耗时较长的原因，对比业务侧与 stage 视角的耗时差异，因该任务分多个 state 可能导致总耗时增加 ","execution_ddl":"无","executor":["无"]},"sentence_id":["37","38","39","40","41","42","43","44","45","46","47","48","49","50","51","52","53","54","55","56","57","58","59","60","61","62","63","64","65","66","67"],"start_time":660790,"todo_idx":46},{"content":"茂哥将章节生成过程中的强依赖 stage 梳理出来并单独做一个面板，以便后续维护","execution_ddl":"无","execution_time":["无"],"executor":["茂哥"],"polished_res":{"content":"**依赖面板整理**：将章节生成过程中的强依赖 stage 梳理出来，单列一个面板，以解决 stage 多导致的混乱问题，便于后续维护","execution_ddl":"无","executor":["茂哥"]},"sentence_id":["37","38","39","40","41","42","43","44","45","46","47","48","49","50","51","52","53","54","55","56","57","58","59","60","61","62","63","64","65","66","67"],"start_time":859000,"todo_idx":62},{"content":"查看是否有被屏蔽的埋点，导致 Todo stage 的耗时没有显示","execution_ddl":"无","execution_time":["无"],"executor":["无"],"polished_res":{"content":"**埋点情况查看**：查看是否有因无人查询被屏蔽的埋点，导致 Todo stage 的耗时未显示\n","execution_ddl":"无","executor":["无"]},"sentence_id":["68","69","70","71","72","73","74","75","76","77","78","79","80","81","82","83","84","85","86","87","88","89","90","91","92","93","94","95","96","97","98","99","100","101","102","103","104","105","106"],"start_time":1174810,"todo_idx":89},{"content":"梳理录制和 SDK 的 SOP 流程，以补充预案的不足","execution_ddl":"无","execution_time":["无"],"executor":["无"],"polished_res":{"content":"**预案 SOP 梳理**：梳理录制和 SDK 的 SOP 流程，以补充预案的不足，9 月 4 号完成发布会重保相关事项 ","execution_ddl":"无","executor":["无"]},"sentence_id":["141","142","143"],"start_time":1755218,"todo_idx":141}],"transition":[{"label":3,"sentence_id":"0"},{"label":3,"sentence_id":"1"},{"label":1,"sentence_id":"2"},{"label":3,"sentence_id":"3"},{"label":1,"sentence_id":"4"},{"label":3,"sentence_id":"5"},{"label":3,"sentence_id":"6"},{"label":3,"sentence_id":"7"},{"label":3,"sentence_id":"8"},{"label":3,"sentence_id":"9"},{"label":3,"sentence_id":"10"},{"label":3,"sentence_id":"11"},{"label":3,"sentence_id":"12"},{"label":3,"sentence_id":"13"},{"label":3,"sentence_id":"14"},{"label":3,"sentence_id":"15"},{"label":3,"sentence_id":"16"},{"label":3,"sentence_id":"17"},{"label":3,"sentence_id":"18"},{"label":3,"sentence_id":"19"},{"label":3,"sentence_id":"20"},{"label":3,"sentence_id":"21"},{"label":3,"sentence_id":"22"},{"label":3,"sentence_id":"23"},{"label":3,"sentence_id":"24"},{"label":3,"sentence_id":"25"},{"label":3,"sentence_id":"26"},{"label":3,"sentence_id":"27"},{"label":3,"sentence_id":"28"},{"label":3,"sentence_id":"29"},{"label":3,"sentence_id":"30"},{"label":3,"sentence_id":"31"},{"label":3,"sentence_id":"32"},{"label":3,"sentence_id":"33"},{"label":3,"sentence_id":"34"},{"label":3,"sentence_id":"35"},{"label":3,"sentence_id":"36"},{"label":3,"sentence_id":"37"},{"label":3,"sentence_id":"38"},{"label":3,"sentence_id":"39"},{"label":3,"sentence_id":"40"},{"label":3,"sentence_id":"41"},{"label":3,"sentence_id":"42"},{"label":3,"sentence_id":"43"},{"label":3,"sentence_id":"44"},{"label":3,"sentence_id":"45"},{"label":3,"sentence_id":"46"},{"label":3,"sentence_id":"47"},{"label":3,"sentence_id":"48"},{"label":3,"sentence_id":"49"},{"label":3,"sentence_id":"50"},{"label":3,"sentence_id":"51"},{"label":3,"sentence_id":"52"},{"label":3,"sentence_id":"53"},{"label":3,"sentence_id":"54"},{"label":3,"sentence_id":"55"},{"label":3,"sentence_id":"56"},{"label":3,"sentence_id":"57"},{"label":3,"sentence_id":"58"},{"label":3,"sentence_id":"59"},{"label":3,"sentence_id":"60"},{"label":3,"sentence_id":"61"},{"label":3,"sentence_id":"62"},{"label":3,"sentence_id":"63"},{"label":3,"sentence_id":"64"},{"label":3,"sentence_id":"65"},{"label":3,"sentence_id":"66"},{"label":3,"sentence_id":"67"},{"label":3,"sentence_id":"68"},{"label":3,"sentence_id":"69"},{"label":3,"sentence_id":"70"},{"label":3,"sentence_id":"71"},{"label":3,"sentence_id":"72"},{"label":3,"sentence_id":"73"},{"label":3,"sentence_id":"74"},{"label":3,"sentence_id":"75"},{"label":3,"sentence_id":"76"},{"label":3,"sentence_id":"77"},{"label":3,"sentence_id":"78"},{"label":1,"sentence_id":"79"},{"label":3,"sentence_id":"80"},{"label":3,"sentence_id":"81"},{"label":3,"sentence_id":"82"},{"label":3,"sentence_id":"83"},{"label":3,"sentence_id":"84"},{"label":3,"sentence_id":"85"},{"label":3,"sentence_id":"86"},{"label":3,"sentence_id":"87"},{"label":3,"sentence_id":"88"},{"label":3,"sentence_id":"89"},{"label":3,"sentence_id":"90"},{"label":3,"sentence_id":"91"},{"label":3,"sentence_id":"92"},{"label":1,"sentence_id":"93"},{"label":3,"sentence_id":"94"},{"label":3,"sentence_id":"95"},{"label":3,"sentence_id":"96"},{"label":3,"sentence_id":"97"},{"label":3,"sentence_id":"98"},{"label":3,"sentence_id":"99"},{"label":3,"sentence_id":"100"},{"label":1,"sentence_id":"101"},{"label":3,"sentence_id":"102"},{"label":1,"sentence_id":"103"},{"label":1,"sentence_id":"104"},{"label":3,"sentence_id":"105"},{"label":3,"sentence_id":"106"},{"label":3,"sentence_id":"107"},{"label":3,"sentence_id":"108"},{"label":3,"sentence_id":"109"},{"label":3,"sentence_id":"110"},{"label":3,"sentence_id":"111"},{"label":3,"sentence_id":"112"},{"label":3,"sentence_id":"113"},{"label":3,"sentence_id":"114"},{"label":3,"sentence_id":"115"},{"label":3,"sentence_id":"116"},{"label":3,"sentence_id":"117"},{"label":3,"sentence_id":"118"},{"label":3,"sentence_id":"119"},{"label":3,"sentence_id":"120"},{"label":3,"sentence_id":"121"},{"label":3,"sentence_id":"122"},{"label":3,"sentence_id":"123"},{"label":3,"sentence_id":"124"},{"label":3,"sentence_id":"125"},{"label":3,"sentence_id":"126"},{"label":3,"sentence_id":"127"},{"label":3,"sentence_id":"128"},{"label":3,"sentence_id":"129"},{"label":3,"sentence_id":"130"},{"label":3,"sentence_id":"131"},{"label":3,"sentence_id":"132"},{"label":3,"sentence_id":"133"},{"label":3,"sentence_id":"134"},{"label":3,"sentence_id":"135"},{"label":3,"sentence_id":"136"},{"label":3,"sentence_id":"137"},{"label":3,"sentence_id":"138"},{"label":3,"sentence_id":"139"},{"label":3,"sentence_id":"140"},{"label":3,"sentence_id":"141"},{"label":3,"sentence_id":"142"},{"label":3,"sentence_id":"143"}]}
```

<span id="e9094db8"></span>
###  全文总结 SummarizationFile
```JSON
{
    "title":""      // 标题，
    "paragraph":""  //全文总结
}
```

```Plain Text
{
    "paragraph": "会议讨论了监控大盘各部分监控情况及纪要兜底方案，还提及部分未确定事项，具体如下：\n1. **监控大盘总体情况** ：\n  - **整体结构** ：监控大盘分为总体监控、纪要相关、Todo相关、章节相关、segment任务相关五个部分。\n  - **业务侧监控** ：关注AI任务提交、堆积、等待时间等，可设置标准线用于告警，关注任务失败、生成时间等情况。\n  - **达芬奇视角监控** ：展示任务运行数量、失败状态码等，比业务侧更细致，各监控遵循业务侧、达芬奇侧、INFO call侧三层原则。\n2. **各部分监控细节** ：\n  - **纪要章节和Todo** ：业务侧监控关注提交任务失败重试和耗时；达芬奇视角分为多个stage，关注任务运行数量、失败情况及耗时；INFO call视角目前只有Pre点，后续周四上线online点。\n  - **章节监控** ：业务侧关注提交任务失败和耗时；达芬奇视角stage多，有部分stage报错需关注；INFO call主要关注additional gene等环节。\n  - **segment任务** ：关注任务提交、异常退出、运行数量及各阶段处理耗时，有部分任务重试5次后放弃，需找AI人员查看原因。\n3. **单个任务监控** ：将各环节处理失败埋点展示，填入meeting ID可查看具体环节失败情况、各stage埋点及耗时等，info call调LPP报错也会显示。\n4. **纪要兜底方案** ：针对总结、待办、章节可能出现的异常情况，如发起失败、处理失败、卡死无callback等，提出标记成功、使用TC配置兜底、手动模拟callback等处理方式，编辑场景也有相应处理规则。\n5. **未确定事项** ：发布会展示链接使用文档还是妙记未确定，展示内容使用会中还是会后的也未确定，需后续确认。\n6. **其他事项** ：AI大模型资源扩容可满足申请诉求；需梳理录制和SDK的发布会重保预案；发布会重保时间确定为9月4号。",
    "title": "会议主题：监控大盘及纪要兜底方案讨论"
}
```

<span id="c8f1e00e"></span>
### 翻译 TranslationFile
<Attachment link="https://p9-arcosite.byteimg.com/tos-cn-i-goo7wpa0wc/cfc648a5c9db4bdd9613c95a408da59c~tplv-goo7wpa0wc-image.image" name="翻译 TranslationFile.rtf" ></Attachment>

<span id="9a974496"></span>
## 错误码
对应X-Api-Status-Code，当发生异常时具体原因详细见X-Api-Message，另外X-Tt-Logid是重要排查标记
可能遇到的问题：

1. QPS限流：请按照要求控制并发，超出将驳回可重试
2. 文件不符合规范：大小1G、时长2小时
3. 传参不符合诉求：注意必选和可选的组合方式
4. 提交接口超时：5s timeout，超时可尝试重试
5. 文件URL超时： 24小时有效期 


| | | | \
|错误码 |含义 |说明 |
|---|---|---|
| | | | \
|20000000 |成功 | |
| | | | \
|20000001 |正在处理中 | |
| | | | \
|20000002 |任务在队列中 | |
| | | | \
|20000003 |静音音频 |返回该错误码无需重新query，直接重新submit |
| | | | \
|45000001 |请求参数无效 |请求参数缺失必需字段 / 字段值无效 / 重复请求。 |
| | | | \
|45000002 |空音频 | |
| | | | \
|45000151 |音频格式不正确 | |
| | | | \
|550xxxx |服务内部处理错误 | |
| | | | \
|55000031 |服务器繁忙 |服务过载，无法处理当前请求。 |


<span id="479a9fc9"></span>
## goDemo
<Attachment link="https://p9-arcosite.byteimg.com/tos-cn-i-goo7wpa0wc/40fac713c67246acad1b67728db448a4~tplv-goo7wpa0wc-image.image" name="lark_go.zip" ></Attachment>
<span id="15cf3f38"></span>
## javaDemo
<Attachment link="https://p9-arcosite.byteimg.com/tos-cn-i-goo7wpa0wc/bfc20a12bb9046f29191cbafa66a1739~tplv-goo7wpa0wc-image.image" name="lark.zip" ></Attachment>
<span id="0b03e55d"></span>
## pythonDemo
<Attachment link="https://p9-arcosite.byteimg.com/tos-cn-i-goo7wpa0wc/ad038e6fdf3f48d9bb2aca29049c302f~tplv-goo7wpa0wc-image.image" name="lark_python.zip" ></Attachment>


