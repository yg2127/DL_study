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
    

```
git checkout main
git pull origin main
git checkout 본인브랜치
git merge main   # 혹은 rebase도 가능
```
