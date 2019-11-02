from django.contrib import admin
from .models import Question

# 自定义后台表单
class QuestionAdmin(admin.ModelAdmin):  # 创建一个模型后台类
    # fields = ['pub_date','question_text']   # 后台管理按照这个list顺序显示字段内容
    # 将表单分为几个字段集
    fieldsets = [   # 每个字段也有先后顺序
        ('Date information',{'fields':['pub_date']}),
        (None,{'fields':['question_text']}),    # 第一个元素是字段集的标题
    ]

admin.site.register(Question,QuestionAdmin)

