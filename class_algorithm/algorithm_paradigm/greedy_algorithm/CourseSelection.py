def course_selection(course_list):
    # 코드를 작성하세요.
    course_list.sort(key=lambda x:x[1])
    # course_list.sort(key=lambda x:(x[0],-x[1])) # 첫 번째 원소는 오름차순, 두 번째 원소는 내림차순 정렬

    course_selected = []
    end = 0
    for course in course_list:
        if course[0] > end:
            end = course[1]
            course_selected.append(course)

    return course_selected


# 테스트
print(course_selection([(6, 10), (2, 3), (4, 5), (1, 7), (6, 8), (9, 10)]))
print(course_selection([(1, 2), (3, 4), (0, 6), (5, 7), (8, 9), (5, 9)]))
print(course_selection([(4, 7), (2, 5), (1, 3), (8, 10), (5, 9), (2, 5), (13, 16), (9, 11), (1, 8)]))
