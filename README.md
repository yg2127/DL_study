# DL_study
브랜치 분기 가이드

## **✅ 팀원 브랜치 분기 매뉴얼 (main → 개인 브랜치)**

1️⃣ 먼저 GitHub에서 프로젝트 클론

```
git clone https://github.com/yg2127/DL_study.git
cd DL_study
```

2️⃣ main 브랜치로 전환하고 최신 상태로 유지

```
git checkout main
git pull origin main
```

3️⃣ 개인 브랜치 생성 및 GitHub에 등록

```
git checkout -b 본인이름   # 예: minji
git push -u origin 본인이름
```

- -b는 브랜치 생성
    
- -u는 원격(origin) 브랜치와 연결 (tracking 설정)
    

---

## **🧠 협업 팁**

- **main에는 직접 push 금지!** → 오직 PR로 병합
    
- 작업 완료되면 GitHub에서 **Pull Request (PR)** 열기
    
- 충돌 방지를 위해 작업 전 꼭 main에서 pull 하고 시작

- 각자의 이름의 폴더에서 코드 실행 하기 (충돌 방지 위함)

- 같은 파일의 같은 줄 수정시 주의 요망 (다른 사람꺼 코드 맘대로 수정하면 안됨!)
    

```
git checkout main
git pull origin main
git checkout 본인브랜치
git merge main   # 혹은 rebase도 가능
```

브랜치 분기 가이드

## **✅ 팀원 브랜치 분기 매뉴얼 (main → 개인 브랜치)**

1️⃣ 먼저 GitHub에서 프로젝트 클론

```
git clone https://github.com/yg2127/DL_study.git
cd DL_study
```

2️⃣ main 브랜치로 전환하고 최신 상태로 유지

```
git checkout main 
git pull origin main
```

3️⃣ 개인 브랜치 생성 및 GitHub에 등록

```
git checkout -b 본인이름   # 예: minji
git push -u origin 본인이름
```

- -b는 브랜치 생성

- -u는 원격(origin) 브랜치와 연결 (tracking 설정)


---

## **🧠 협업 팁**

- **main에는 직접 push 금지!** → 오직 PR로 병합

- 작업 완료되면 GitHub에서 **Pull Request (PR)** 열기

- 충돌 방지를 위해 작업 전 꼭 main에서 pull 하고 시작


```
git checkout main
git pull origin main
git checkout 본인브랜치
git merge main   # 혹은 rebase도 가능
```
## GitHub basic concept

pull = 해당 브랜치를 내 로컬라포로 불러오고 내 온라인 브랜치까지 덮어쓰게 하는것 (main에 대한 변경사항만, 내 브랜치에 있는 정보들은 저장되어있음)

fetch = 해당 온라인 브랜치의 변경사항을 가져오기만 함 내 (온/오프라인) 브랜치에 영향없음!

merge = 현재 브랜치에서 다른 브랜치의 변경사항을 병합하는것

fetch + merge = pull!

commit = 변경사항을 저장 (version 저장)

push = 내 변경사항을 온라인 라포에 업로드하는 것

branch = 브랜치별로 각각 다른 파일을 업로드할 수 있는 것

rebase = 한 브랜치의 커밋들을 다른 브랜치 뒤에 “순차적으로 재적용”시켜서 깔끔한 선형(Linear) 히스토리를 만든다. 히스토리를 “되돌리기 어려운” 방식으로 수정 - 이미 공유된 브랜치엔 위험할 수 있음
