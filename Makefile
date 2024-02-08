
## TESTS ##
api_test:
	uvicorn api.enhancer_api:app --reload &
	sleep 5
	streamlit run test_api.py
