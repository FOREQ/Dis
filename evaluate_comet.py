from evaluate import load

comet_metric = load('comet')
source = ["Статья 187 с изменениями, внесёнными законами РК от 07.11.2014 № 248-V (вводится в действие с 01.01.2015); от 13.11.2015 № 400-V (вводится в действие по истечении десяти календарных дней после дня его первого официального опубликования); от 24.11.2015 № 419-V (вводится в действие с 01.01.2016); от 24.11.2015 № 422-V (вводится в действие с 01.01.2016); от 22.12.2016 № 28-VІ (вводится в действие по истечении десяти календарных дней после дня его первого официального опубликования); от 03.07.2017 № 84-VI (вводится в действие по истечении десяти календарных дней после дня его первого официального опубликования); от 28.12.2017 № 128-VI (вводится в действие по истечении десяти календарных дней после дня его первого официального опубликования); от 12.07.2018 № 180-VI (вводится в действие по истечении десяти календарных дней после дня его первого официального опубликования); от 21.01.2019 № 217-VI (вводится в действие по истечении десяти календарных дней после дня его первого официального опубликования); от 01.04.2019 № 240-VI (вводится в действие по истечении десяти календарных дней после дня его первого официального опубликования); от 28.10.2019 № 268-VI (вводится в действие по истечении десяти календарных дней после дня его первого официального опубликования); от 27.12.2019 № 290-VІ (вводится в действие по истечении десяти календарных дней после дня его первого официального опубликования); от 27.12.2019 № 292-VІ (порядок введения в действие см.ст.2); от 25.05.2020 № 332-VI (вводится в действие по истечении десяти календарных дней после дня его первого официального опубликования); от 06.10.2020 № 365-VI (вводится в действие по истечении десяти календарных дней после дня его первого официального опубликования)."]
hypothesis = ["187-бапқа өзгеріс енгізілді-ҚР 07.11.2014 № 248-V (01.01.2015 бастап қолданысқа енгізіледі); 13.11.2015 № 400-V (алғашқы ресми жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі); 24.11.2015 № 419-V (01.01.2016 бастап қолданысқа енгізіледі) заңдарымен. 24.11.2015 № 422-V (01.01.2016 бастап қолданысқа енгізіледі); 22.12.2016 № 28-VІ (алғашқы ресми жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі); 03.07.2017 № 84-VI (алғашқы ресми жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі); 28.12.2017 № 128-VI (алғашқы ресми жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі); 12.07.2018 № 180-VI (күнтізбелік он күн өткен соң қолданысқа енгізіледі 21.01.2019 № 217-VI (алғашқы ресми жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі); ; 01.04.2019 № 240-VI (алғашқы ресми жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі); 28.10.2019 № 268-VI (алғашқы ресми жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі); 27.12.2019 № 290-VI (күнтізбелік он күн өткен соң қолданысқа енгізіледі 27.12.2019 № 292-VІ (қолданысқа енгізілу тәртібін 2-баптан қараңыз); 2020.05.ізбелік он күн өткен соң қолданысқа енгізіледі); ; 06.10.2020 № 365-VI (алғашқы ресми жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі)."]
reference = ["187-бапқа өзгерістер енгізілді – ҚР 07.11.2014 № 248-V (01.01.2015 бастап қолданысқа енгізіледі); 13.11.2015 № 400-V (алғашқы ресми жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі); 24.11.2015 № 419-V (01.01.2016 бастап қолданысқа енгізіледі); 24.11.2015 № 422-V (01.01.2016 бастап қолданысқа енгізіледі); 22.12.2016 № 28-VI (алғашқы ресми жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі); 03.07.2017 № 84-VI (алғашқы ресми жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі); 28.12.2017 № 128-VI (алғашқы ресми жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі); 12.07.2018 № 180-VІ (алғашқы ресми жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі); 21.01.2019 № 217-VI (алғашқы ресми жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі); 01.04.2019 № 240-VI (алғашқы ресми жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі); 28.10.2019 № 268-VI (алғашқы ресми жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі); 27.12.2019 № 290-VІ (алғашқы ресми жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі); 27.12.2019 № 292-VІ (қолданысқа енгізілу тәртібін 2-баптан қараңыз); 25.05.2020 № 332-VI (алғашқы ресми жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі); 06.10.2020 № 365-VI (алғашқы ресми жарияланған күнінен кейін күнтізбелік он күн өткен соң қолданысқа енгізіледі)"]

comet_score = comet_metric.compute(predictions=hypothesis, references=reference, sources=source)

print(comet_score)