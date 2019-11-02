from django.db import models
import datetime
from django.utils import timezone

# 模型代码给了 Django 很多信息，通过这些信息，Django 可以：
# 为这个应用创建数据库 schema（生成 CREATE TABLE 语句）。
# 创建可以与 Question 和 Choice 对象进行交互的 Python 数据库 API。
class Question(models.Model):
    # 字段：类变量
    # 每个 Field 类实例变量的名字，是字段名，也就是表列名
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')   # 定义了 人类可读的名字

    def was_published_recently(self):
        return timezone.now() >= self.pub_date >= (timezone.now() - datetime.timedelta(days=1))

    def __str__(self):
        return self.question_text

class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)

    def __str__(self):
        return self.choice_text
