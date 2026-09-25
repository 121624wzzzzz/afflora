# Fixed qualitative panel

Post-fit descriptive diagnostics only. For each new model/task, compare the first pre-specified seed of stack and budget LoRA. Select the first two exact-content wins and losses in fixed test order, if available. These are illustrative outcomes, not a random estimate of quality.

## cluener / qwen25_3b_base / seed 6100 / win / cluener_dev_1289

Input: 中国人民大学行政管理学研究所所长毛寿龙说，超过20万元每平米的补偿要求确实有些高。

Gold: `[{"occurrence":0,"text":"中国人民大学行政管理学研究所","type":"organization"},{"occurrence":0,"text":"所长","type":"position"},{"occurrence":0,"text":"毛寿龙","type":"name"}]`

Budget LoRA:
```text
[{"occurrence":0,"text":"中国人民大学","type":"organization"},{"occurrence":0,"text":"行政管理学研究所","type":"organization"},{"occurrence":0,"text":"所长","type":"position"},{"occurrence":0,"text":"毛寿龙","type":"name"}]
```

LoRA + A-LoRA:
```text
[{"occurrence":0,"text":"中国人民大学行政管理学研究所","type":"organization"},{"occurrence":0,"text":"所长","type":"position"},{"occurrence":0,"text":"毛寿龙","type":"name"}]
```

## cluener / qwen25_3b_base / seed 6100 / win / cluener_dev_829

Input: 火来胜：一方好砚就如同一个正派人

Gold: `[{"occurrence":0,"text":"火来胜","type":"name"}]`

Budget LoRA:
```text
[{"occurrence":0,"text":"火来胜","type":"name"},{"occurrence":0,"text":"一方好砚","type":"book"}]
```

LoRA + A-LoRA:
```text
[{"occurrence":0,"text":"火来胜","type":"name"}]
```

## cluener / qwen25_3b_base / seed 6100 / loss / cluener_dev_495

Input: 【赛迪网讯】7月1日，据悉，网易于昨日发内部邮件，邀请全体员工测试《魔兽世界》(下称“

Gold: `[{"occurrence":0,"text":"网易","type":"company"},{"occurrence":0,"text":"《魔兽世界》","type":"game"}]`

Budget LoRA:
```text
[{"occurrence":0,"text":"网易","type":"company"},{"occurrence":0,"text":"《魔兽世界》","type":"game"}]
```

LoRA + A-LoRA:
```text
[{"occurrence":0,"text":"网易","type":"company"},{"occurrence":0,"text":"《魔兽世界》(下称“","type":"game"}]
```

## cluener / qwen25_3b_base / seed 6100 / loss / cluener_dev_93

Input: 中国内地邮票由原来国家邮政局改成由现在的中国邮政集团公司印制发行，从此政企分开，

Gold: `[{"occurrence":0,"text":"国家邮政局","type":"government"},{"occurrence":0,"text":"中国邮政集团公司","type":"company"}]`

Budget LoRA:
```text
[{"occurrence":0,"text":"国家邮政局","type":"government"},{"occurrence":0,"text":"中国邮政集团公司","type":"company"}]
```

LoRA + A-LoRA:
```text
[{"occurrence":0,"text":"中国内地邮票","type":"book"},{"occurrence":0,"text":"国家邮政局","type":"government"},{"occurrence":0,"text":"中国邮政集团公司","type":"company"}]
```

## cluener / qwen25_7b_base / seed 6100 / win / cluener_dev_955

Input: 他们的本意是要迎接暴雪《星际争霸2》的Beta测试开放。但是，小编也不清楚是暴雪跳票了，

Gold: `[{"occurrence":0,"text":"暴雪","type":"company"},{"occurrence":0,"text":"《星际争霸2》","type":"game"},{"occurrence":1,"text":"暴雪","type":"company"}]`

Budget LoRA:
```text
[{"occurrence":0,"text":"暴雪","type":"company"},{"occurrence":0,"text":"《星际争霸2》","type":"game"}]
```

LoRA + A-LoRA:
```text
[{"occurrence":0,"text":"暴雪","type":"company"},{"occurrence":0,"text":"《星际争霸2》","type":"game"},{"occurrence":1,"text":"暴雪","type":"company"}]
```

## cluener / qwen25_7b_base / seed 6100 / win / cluener_dev_659

Input: 美国军方于美国时间17日推出免费军事射击游戏《美国陆军3（America's

Gold: `[{"occurrence":0,"text":"美国军方","type":"government"},{"occurrence":0,"text":"《美国陆军3（America's","type":"game"}]`

Budget LoRA:
```text
[{"occurrence":0,"text":"美国军方","type":"government"},{"occurrence":0,"text":"美国","type":"address"},{"occurrence":0,"text":"《美国陆军3（America's","type":"game"}]
```

LoRA + A-LoRA:
```text
[{"occurrence":0,"text":"美国军方","type":"government"},{"occurrence":0,"text":"《美国陆军3（America's","type":"game"}]
```

## cluener / qwen25_7b_base / seed 6100 / loss / cluener_dev_1186

Input: 红御营销总监王婷鸿；伟业顾问策划总监李宗诺，还有我们新浪乐居主编杨熙，欢迎大家的到来！

Gold: `[{"occurrence":0,"text":"红御","type":"company"},{"occurrence":0,"text":"营销总监","type":"position"},{"occurrence":0,"text":"王婷鸿","type":"name"},{"occurrence":0,"text":"伟业顾问","type":"company"},{"occurrence":0,"text":"策划总监","type":"position"},{"occurrence":0,"text":"李宗诺","type":"name"},{"occurrence":0,"text":"新浪乐居","type":"company"},{"occurrence":0,"text":"主编","type":"position"},{"occurrence":0,"text":"杨熙","type":"name"}]`

Budget LoRA:
```text
[{"occurrence":0,"text":"红御","type":"company"},{"occurrence":0,"text":"营销总监","type":"position"},{"occurrence":0,"text":"王婷鸿","type":"name"},{"occurrence":0,"text":"伟业顾问","type":"company"},{"occurrence":0,"text":"策划总监","type":"position"},{"occurrence":0,"text":"李宗诺","type":"name"},{"occurrence":0,"text":"新浪乐居","type":"company"},{"occurrence":0,"text":"主编","type":"position"},{"occurrence":0,"text":"杨熙","type":"name"}]
```

LoRA + A-LoRA:
```text
[{"occurrence":0,"text":"红御","type":"company"},{"occurrence":0,"text":"总监","type":"position"},{"occurrence":0,"text":"王婷鸿","type":"name"},{"occurrence":0,"text":"伟业顾问","type":"company"},{"occurrence":0,"text":"总监","type":"position"},{"occurrence":0,"text":"李宗诺","type":"name"},{"occurrence":0,"text":"主编","type":"position"},{"occurrence":0,"text":"杨熙","type":"name"},{"occurrence":0,"text":"新浪乐居","type":"company"}]
```

## cluener / qwen25_7b_base / seed 6100 / loss / cluener_dev_801

Input: 当年威尼斯的一些著名的场所都会在游戏中出现，比如达芬奇的工作间、圣马可大教堂、

Gold: `[{"occurrence":0,"text":"威尼斯","type":"address"},{"occurrence":0,"text":"达芬奇","type":"name"},{"occurrence":0,"text":"圣马可大教堂","type":"scene"}]`

Budget LoRA:
```text
[{"occurrence":0,"text":"威尼斯","type":"address"},{"occurrence":0,"text":"达芬奇","type":"name"},{"occurrence":0,"text":"圣马可大教堂","type":"scene"}]
```

LoRA + A-LoRA:
```text
[{"occurrence":0,"text":"威尼斯","type":"address"},{"occurrence":0,"text":"达芬奇的工作间","type":"scene"},{"occurrence":0,"text":"圣马可大教堂","type":"scene"}]
```

## wikisql / qwen25_3b_base / seed 7100 / win / wikisql-test-995

Input: How many places of whatever size have a 2007 population estimation of 7996?; columns: ["Name","Population (2002 census)","Population (2007 estimation)","Population (2011 census)","Area (km²)","Density (pop/km²)"]

Gold: `{"agg":3,"conds":[[2,0,7996]],"sel":4}`

Budget LoRA:
```text
{"agg":3,"conds":[[4,0,"whatever"],[2,0,7996]],"sel":0}
```

LoRA + A-LoRA:
```text
{"agg":3,"conds":[[2,0,7996]],"sel":0}
```

## wikisql / qwen25_3b_base / seed 7100 / win / wikisql-test-14566

Input: What is the highest number born in a none EU state (1000) from Denmark with a total population (1000) less than 5,534?; columns: ["Country","Total population (1000)","Total Foreign-born (1000)","Born in other EU state (1000)","Born in a non EU state (1000)"]

Gold: `{"agg":1,"conds":[[0,0,"denmark"],[1,2,"5,534"]],"sel":4}`

Budget LoRA:
```text
{"agg":1,"conds":[[0,0,"denmark"],[1,2,5,534]],"sel":4}
```

LoRA + A-LoRA:
```text
{"agg":1,"conds":[[0,0,"denmark"],[1,2,5534]],"sel":4}
```

## wikisql / qwen25_3b_base / seed 7100 / loss / wikisql-test-4260

Input: What is the chromosomal location of il-1β?; columns: ["Name","Family name","Receptor","Coreceptor","Property","Chromosomal location"]

Gold: `{"agg":0,"conds":[[0,0,"IL-1β"]],"sel":5}`

Budget LoRA:
```text
{"agg":0,"conds":[[0,0,"Il-1β"]],"sel":5}
```

LoRA + A-LoRA:
```text
{"agg":0,"conds":[[2,0,"IL-1β"]],"sel":5}
```

## wikisql / qwen25_3b_base / seed 7100 / loss / wikisql-test-6180

Input: Which Population is the lowest one that has a Median family income of $74,905, and a Number of households smaller than 145,790?; columns: ["County","Per capita income","Median household income","Median family income","Population","Number of households"]

Gold: `{"agg":2,"conds":[[3,0,"$74,905"],[5,2,"145,790"]],"sel":4}`

Budget LoRA:
```text
{"agg":2,"conds":[[3,0,"$74,905"],[5,2,145790]],"sel":4}
```

LoRA + A-LoRA:
```text
{"agg":2,"conds":[[3,0,"$74,905"],[5,2,145,790]],"sel":4}
```

## wikisql / qwen25_7b_base / seed 7100 / win / wikisql-test-11915

Input: Who wrote the title "Parting Glances" ?; columns: ["No. in series","No. in season","Title","Director","Writer(s)","Original air date","Production code"]

Gold: `{"agg":0,"conds":[[2,0,"\"parting glances\""]],"sel":4}`

Budget LoRA:
```text
{"agg":0,"conds":[[2,0,"parting glances"]],"sel":4}
```

LoRA + A-LoRA:
```text
{"agg":0,"conds":[[2,0,"\"parting glances\""]],"sel":4}
```

## wikisql / qwen25_7b_base / seed 7100 / win / wikisql-test-2434

Input: where births is 388 000 how many number is natural increase; columns: ["Period","Births","Deaths","Difference","Crude Birth Rate","Crude Death Rate","Natural Increase","Total Fertility Rate","Infant Mortality Rate"]

Gold: `{"agg":3,"conds":[[1,0,"388 000"]],"sel":6}`

Budget LoRA:
```text
{"agg":0,"conds":[[1,0,"388 000"]],"sel":6}
```

LoRA + A-LoRA:
```text
{"agg":3,"conds":[[1,0,"388 000"]],"sel":6}
```

## wikisql / qwen25_7b_base / seed 7100 / loss / wikisql-test-2724

Input: Name the conference player of the year for east tennessee state; columns: ["Conference","Regular Season Winner","Conference Player of the Year","Conference Tournament","Tournament Venue (City)","Tournament Winner"]

Gold: `{"agg":0,"conds":[[1,0,"East Tennessee State"]],"sel":2}`

Budget LoRA:
```text
{"agg":0,"conds":[[1,0,"east tennessee state"]],"sel":2}
```

LoRA + A-LoRA:
```text
{"agg":0,"conds":[[0,0,"east tennessee state"]],"sel":2}
```

## wikisql / qwen25_7b_base / seed 7100 / loss / wikisql-test-3067

Input: Name the language for b. ajith kumar; columns: ["Name of Award","Name of Film","Language","Awardee(s)","Cash Prize"]

Gold: `{"agg":0,"conds":[[3,0,"B. Ajith Kumar"]],"sel":2}`

Budget LoRA:
```text
{"agg":0,"conds":[[3,0,"b. ajith kumar"]],"sel":2}
```

LoRA + A-LoRA:
```text
{"agg":0,"conds":[[1,0,"b. ajith kumar"]],"sel":2}
```
